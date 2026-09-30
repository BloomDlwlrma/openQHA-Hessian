"""The HVP by autograd on the REAL engine
equals the shipped full Hessian, column by column.

INTEGRATION. Loads MACE-OFF23_medium, takes the reference geometry of propanal basin 0
(the `.hess` of tests/data/propanal_molecule, as t_hessian_compare_engine does) and
compares `openqha_hessian.hvp.hvp_from_atoms` with the 30 Cartesian unit probes against
`calc.get_hessian()` reshaped (3N, 3N): T03 section 10 measured 1.4e-14 eV/A^2 on one
column; the bound here is 1e-12 on all of them. Then: `training=True` returns a tensor
with a graph and `training=False` one without; a two-molecule batch (propanal twice, the
second shifted 10 A) returns each molecule's own HVP to 1e-12; `along_mode_curvature`
on the reference modes equals `L^T K L` from the full matrix to 1e-12. SKIPs without the
model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tests/, where _testlib lives
from _testlib import openqha_src                                # noqa: E402

SRC = openqha_src() / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    import numpy as np
    import torch
    from ase import Atoms
    from openqha.potentials import engine
    from openqha.qm_interfaces import orca
    from openqha.store import layout
    from openqha.thermochem import hessian as hessian_mod
    from openqha_hessian import hvp
    try:
        calc, name, prov = engine.calculator(device="cpu")
    except Exception as exc:                                   # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0
    print("  engine {}  mace {}  fork {}{}".format(name, prov["mace_torch_version"], prov["mace_fork_commit"][:12],
                                                   "  DIRTY" if prov.get("mace_fork_dirty") else ""))
    parsed = orca.parse_hess(layout.orca_level_file(SRC, LEVEL, 0, ".hess"))
    atoms = Atoms(symbols=parsed["symbols"], positions=np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM)
    n = len(atoms); n3 = 3 * n

    # --- the 3N Cartesian unit probes reproduce get_hessian column by column -----------
    H = np.asarray(calc.get_hessian(atoms)).reshape(n3, n3)
    probes = np.eye(n3).reshape(n3, n, 3)
    hv = hvp.hvp_from_atoms(calc, atoms, probes).reshape(n3, n3)      # row j = H e_j = column j of H
    d = float(np.abs(hv - H.T).max())
    check("HVP with the {} unit probes = get_hessian() columns to 1e-12 eV/A^2 (max {:.1e})".format(n3, d), d < 1e-12, d)
    check("get_hessian is symmetric to 1e-10 (so column = row)", float(np.abs(H - H.T).max()) < 1e-10)

    # --- graph or no graph -----------------------------------------------------------------
    model = calc.models[0]
    params = list(model.parameters())
    frozen = [q.requires_grad for q in params]          # the calculator freezes them for inference
    for q in params:
        q.requires_grad_(True)
    bd = calc._clone_batch(calc._atoms_to_batch(atoms)).to_dict()
    p = torch.as_tensor(probes[:2], dtype=torch.get_default_dtype())
    t_tr = hvp.hessian_vector_products(model, bd, p, training=True)
    check("training=True: the HVP carries a graph (grad_fn)", t_tr.grad_fn is not None)
    g = torch.autograd.grad(t_tr.pow(2).sum(), params, allow_unused=True)
    nz = sum(1 for gi in g if gi is not None and float(gi.abs().max()) > 0)
    check("... reaching the parameters ({} of {} tensors with a non-zero gradient)".format(nz, len(params)), nz > 0)
    for q, f in zip(params, frozen):
        q.requires_grad_(f)
    bd = calc._clone_batch(calc._atoms_to_batch(atoms)).to_dict()
    t_inf = hvp.hessian_vector_products(model, bd, p, training=False)
    check("training=False: no grad_fn", t_inf.grad_fn is None)
    check("both modes give the same numbers", float((t_tr.detach() - t_inf).abs().max()) < 1e-12)

    # --- a two-molecule batch is block-diagonal ---------------------------------------------
    two = atoms.copy(); two.positions += 10.0
    pair = atoms + two
    v = np.random.default_rng(0).standard_normal((3, 2 * n, 3))
    hv_pair = hvp.hvp_from_atoms(calc, pair, v)
    hv_a = hvp.hvp_from_atoms(calc, atoms, v[:, :n])
    hv_b = hvp.hvp_from_atoms(calc, two, v[:, n:])
    d = max(float(np.abs(hv_pair[:, :n] - hv_a).max()), float(np.abs(hv_pair[:, n:] - hv_b).max()))
    check("two-molecule batch = per-molecule HVPs to 1e-12 (max {:.1e})".format(d), d < 1e-12, d)

    # --- along-mode curvature without the matrix --------------------------------------------
    masses = atoms.get_masses()
    m = np.repeat(masses, 3)
    K = H / np.sqrt(np.outer(m, m))
    V, _sing, _rank = hessian_mod.rigid_body_vectors(masses, atoms.positions)
    P = np.eye(n3) - V @ V.T
    lam, L = np.linalg.eigh(0.5 * ((P @ K @ P) + (P @ K @ P).T))
    vib = np.linalg.norm(V.T @ L, axis=0) ** 2 < 0.5
    Lv = L[:, vib].T                                             # [n_vib, 3N] mass-weighted unit modes
    D_full = np.einsum("ji,ik,jk->j", Lv, K, Lv)
    D_hvp = hvp.along_mode_curvature(calc, atoms, Lv, masses)
    d = float(np.abs(D_hvp - D_full).max())
    check("along_mode_curvature = L^T K L from the full matrix to 1e-12 (max {:.1e}, {} modes)".format(d, len(D_hvp)),
          d < 1e-12, d)
    w = hessian_mod.eigenvalues_to_cm_inv(D_hvp)
    print("  lowest along-mode curvatures (cm^-1): {}".format(np.round(np.sort(w)[:4], 1)))

    print("\n{} checks, {} failed".format(8, len(FAIL)))
    print("PASS" if not FAIL else "FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
