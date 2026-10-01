"""The Hessian-vector product of a MACE model by automatic differentiation (eq. 9).

PRODUCTION.

    H v = grad_x (grad_x E . v) = -grad_x (F . v)

For a constant probe `v` this is ONE more backward pass over the force graph. It needs
two things and nothing else: forces built with `create_graph=True` -- which is what
mace's forward does whenever it is called with `training=True` -- and the positions
tensor those forces were differentiated against, which is `batch["positions"]` after the
forward (mace sets `requires_grad_` on it in place). So the model's forward is NOT
changed: T03 section 10 took the HVP this way from the unmodified MACE-OFF23_medium graph
and it agreed with a column of the shipped `get_hessian()` to 1.4e-14 eV/A^2. mace has no
`compute_hessian_vector_products` in `mace/modules/utils.py` and its forwards take no
`hessian_probes` argument: this module is where the HVP lives.

Two modes, one function:

  * `create_graph=False` (inference, the ruler, the acceptance tests): the result is a
    plain tensor, the graph is dropped -- memory of one backward pass, whatever `k` is.
  * `create_graph=True` (training, the loss): the result carries a third-order
    graph back to the parameters, so a loss built from it has a gradient (eq. 12).

A batch is block-diagonal: structures in a batch are disconnected graphs, so one backward
pass of `-sum_n F_n . v_n` over the whole batch returns every structure's own `H_n v_n`
(T03 section 8, to 2e-17). That is what makes the cost `(2 + 2k)` forward-equivalents per
batch, against `(2 + 6N)` for the full matrix, and not `k` passes per structure.

Never form `H` here. `MACECalculator.get_hessian` (mace's `compute_hessians_vmap`) is the
full matrix and it is the RULER's path; this is the loss's.

Outside mace, two atoms-level helpers ride on this: `hvp_from_atoms` (one frame) and
`hvp_from_atoms_batch` (a chunk of frames, probes packed as the training batch packs
them -- the balance's estimator uses the latter).
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
    """`H v_j` for Cartesian probes `[k, N, 3]` (or `[k, 3N]`) at `atoms`, as a numpy
    array `[k, N, 3]` in eV/A^2 -- the same columns `get_hessian` carries, outside mace.
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


def hvp_from_atoms_batch(calculator, atoms_list, probes_list):
    """`H v_j` for a CHUNK of frames in one pass (inference) -- the batched twin of
    `hvp_from_atoms`, for the balance's estimator and anything that measures several
    frames at once. Returns a list of `[k_i, N_i, 3]` arrays in eV/A^2, one per frame.

    Each frame's probes `[k_i, 3N_i]` (or `[k_i, N_i, 3]`) are packed, zero outside the
    frame's own slice, into one `[k_max, n_nodes, 3]` tensor -- exactly how the training
    loss packs a batch (`phl_loss`'s `make_probes`) -- one forward builds the force graph
    for the whole batch and one backward per probe index returns every frame's own
    `H_n v_{n,j}` (a batch is block-diagonal; `t_hvp` pins it to 1e-14 and the loss leans
    on it every step). The probes are NOT drawn here: the caller owns the generator and
    the per-frame draws, so the numbers do not depend on the chunking.

    The frames are built and merged with the calculator's own machinery --
    `_atoms_to_batch` per frame (the neighbour list, dtype and device are
    `get_hessian`'s), then one `mace.tools.torch_geometric.Batch.from_data_list`, the
    same collate the training loader does. The per-frame fields are re-wrapped as base
    `Data` objects: mace's `AtomicData` cannot be empty-constructed, so its own
    `to_data_list` cannot split a frame back out, and the fork relies on the base
    class's offset semantics for the merge (edge-index keys take `num_nodes`, the rest
    concatenate). A calculator that pads its batches (`use_compile`) is not supported.
    """
    from mace.tools import torch_geometric

    if len(atoms_list) != len(probes_list):
        raise ValueError("atoms_list and probes_list must pair up: {} frame(s), {} probe set(s)".format(
            len(atoms_list), len(probes_list)))
    model = calculator.models[0]
    data_list = []
    for atoms in atoms_list:
        single = calculator._atoms_to_batch(atoms)
        fields = {k: v for k, v in single.to_dict().items() if k not in ("batch", "ptr")}
        fields["num_nodes"] = len(atoms)
        data_list.append(torch_geometric.Data(**fields))
    bd = calculator._clone_batch(torch_geometric.Batch.from_data_list(data_list)).to_dict()

    prepared, offsets, n_nodes = [], [], 0
    for atoms, probes in zip(atoms_list, probes_list):
        n = len(atoms)
        q = np.asarray(probes, dtype=float).reshape(-1, n, 3)
        prepared.append(q)
        offsets.append(n_nodes)
        n_nodes += n
    k_max = max(q.shape[0] for q in prepared)
    packed = np.zeros((k_max, n_nodes, 3), dtype=float)
    for q, off in zip(prepared, offsets):
        packed[:q.shape[0], off:off + q.shape[1], :] = q
    # the probes take the batch's dtype, exactly as in `hvp_from_atoms`: the calculator
    # runs float64 while the process default stays float32
    pos = bd["positions"]
    pt = torch.as_tensor(packed, dtype=pos.dtype, device=pos.device)
    hv = hessian_vector_products(model, bd, pt, training=False).cpu().numpy()
    return [hv[:q.shape[0], off:off + q.shape[1], :] for q, off in zip(prepared, offsets)]


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
