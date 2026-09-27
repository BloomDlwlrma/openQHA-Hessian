"""The Hessian-vector product of a MACE model by automatic differentiation (eq. 9).

PRODUCTION. Ticket 10 of the Hessian-learning set.

    H v = grad_x (grad_x E . v) = -grad_x (F . v)

For a constant probe `v` this is ONE more backward pass over the force graph. It needs
two things and nothing else: forces built with `create_graph=True` -- which is what
mace's forward does whenever it is called with `training=True` -- and the positions
tensor those forces were differentiated against, which is `batch["positions"]` after the
forward (mace sets `requires_grad_` on it in place). So the model's forward is NOT
changed: T03 section 10 took the HVP this way from the unmodified MACE-OFF23_medium graph
and it agreed with a column of the shipped `get_hessian()` to 1.4e-14 eV/A^2. The design's
`compute_hessian_vector_products` in `mace/modules/utils.py` and the `hessian_probes`
argument of the forwards were therefore never written; this module is where they went.

Two modes, one function:

  * `create_graph=False` (inference, the ruler, the acceptance tests): the result is a
    plain tensor, the graph is dropped -- memory of one backward pass, whatever `k` is.
  * `create_graph=True` (training, ticket 11's loss): the result carries a third-order
    graph back to the parameters, so a loss built from it has a gradient (eq. 12).

A batch is block-diagonal: structures in a batch are disconnected graphs, so one backward
pass of `-sum_n F_n . v_n` over the whole batch returns every structure's own `H_n v_n`
(T03 section 8, to 2e-17). That is what makes the cost `(2 + 2k)` forward-equivalents per
batch, against `(2 + 6N)` for the full matrix, and not `k` passes per structure.

Never form `H` here. `MACECalculator.get_hessian` (mace's `compute_hessians_vmap`) is the
full matrix and it is the RULER's path; this is the loss's.
"""
import numpy as np
import torch


def hvp_from_forces(forces, positions, probes, create_graph=False):
    """`H v_j = -d(F . v_j)/dx` for every probe, on an EXISTING force graph.

    forces      [n_nodes, 3]      built with create_graph=True (mace forward, training=True)
    positions   [n_nodes, 3]      the leaf the forces were differentiated against
    probes      [k, n_nodes, 3]   constant; the whole batch's probes, one row per probe
    returns     [k, n_nodes, 3]   H v_j per structure (the batch is block-diagonal)

    `retain_graph` is always on: the force graph is reused once per probe. A probe that
    touches no differentiable path (`allow_unused`) gives zeros, never None.
    """
    probes = torch.as_tensor(probes, dtype=forces.dtype, device=forces.device)
    if probes.dim() == 2:
        probes = probes.unsqueeze(0)
    if probes.shape[1:] != forces.shape:
        raise ValueError("probes must be [k, n_nodes, 3] = [k, {}]; got {}".format(
            tuple(forces.shape), tuple(probes.shape)))
    out = []
    for j in range(probes.shape[0]):
        g = torch.autograd.grad(
            outputs=[-(forces * probes[j]).sum()],
            inputs=[positions],
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )[0]
        out.append(torch.zeros_like(positions) if g is None else g)
    return torch.stack(out, dim=0)


def hessian_vector_products(model, batch_dict, probes, training=False, compute_stress=False):
    """Forward the model with the force graph kept, then `hvp_from_forces`.

    model       a mace model (`MACECalculator.models[0]`, or the one being trained)
    batch_dict  `batch.to_dict()` of a mace `AtomicData` batch; its `positions` become
                the leaf (mace sets requires_grad in place)
    probes      [k, n_nodes, 3]
    training    True keeps the third-order graph to the parameters (the loss);
                False returns detached values (the ruler)
    returns     [k, n_nodes, 3]

    The forward is called with `training=True` in BOTH cases: that flag only decides
    whether mace builds the forces with `create_graph=True`, which the HVP needs. The
    model has no dropout or batch-norm, so the numbers do not change with it.
    """
    out = model(batch_dict, training=True, compute_force=True, compute_stress=compute_stress)
    forces = out["forces"]
    positions = batch_dict["positions"]
    if forces is None or not positions.requires_grad:
        raise RuntimeError("the forward gave no force graph; the model must be called with compute_force")
    hvp = hvp_from_forces(forces, positions, probes, create_graph=training)
    return hvp if training else hvp.detach()


def hvp_from_atoms(calculator, atoms, probes):
    """The design's `get_hessian_vector_products`, outside mace: `H v_j` for Cartesian
    probes `[k, N, 3]` (or `[k, 3N]`) at `atoms`, as a numpy array `[k, N, 3]` in eV/A^2.
    Inference: no graph is kept. Uses the calculator's own batching so the neighbour
    list, dtype and device are exactly `get_hessian`'s."""
    model = calculator.models[0]
    batch = calculator._atoms_to_batch(atoms)
    bd = calculator._clone_batch(batch).to_dict()
    n = len(atoms)
    p = np.asarray(probes, dtype=float).reshape(-1, n, 3)
    # The probes take the BATCH's dtype, not torch's default: the calculator runs in
    # float64 while the process default stays float32, and a probe rounded to float32
    # costs 1e-7 relative on every product (measured: 8e-7 eV/A^2 on a mode probe,
    # against 1.4e-14 with 0/1 unit probes that float32 holds exactly).
    pos = bd["positions"]
    pt = torch.as_tensor(p, dtype=pos.dtype, device=pos.device)
    hvp = hessian_vector_products(model, bd, pt, training=False)
    return hvp.cpu().numpy()


def along_mode_curvature(calculator, atoms, modes, masses):
    """`D_jj = L_j^T M^-1/2 H M^-1/2 L_j` for mass-weighted unit modes `L` `[m, 3N]`
    (family 4's diagonal, `hessian_compare`'s OMEGA_ALONG_REF_CM squared, in
    eV A^-2 amu^-1) WITHOUT forming `H`: one HVP per mode with the probe
    `v_j = M^-1/2 L_j`, then `D_jj = v_j . (H v_j)`."""
    n = len(atoms)
    L = np.asarray(modes, dtype=float).reshape(-1, 3 * n)
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(np.asarray(masses, dtype=float), 3))
    probes = (L * inv_sqrt_m[None, :]).reshape(-1, n, 3)
    hv = hvp_from_atoms(calculator, atoms, probes).reshape(-1, 3 * n)
    return np.einsum("ji,ji->j", probes.reshape(-1, 3 * n), hv)
