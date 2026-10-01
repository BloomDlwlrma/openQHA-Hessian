"""The fit set, the epoch-0 balance (exact and estimator) and the replay
arithmetic -- no engine (a fake calculator answers with stored Hessians and a quadratic
toy answers the HVP protocol with the same curvature).

Asserted: `build_fit_dataset` re-splits labelled frames by frame, marks `PURPOSE = fit`
and writes the merged file; `epoch_zero_balance` returns the three terms of eq. 11 on the
base model and `w_H = w_F L_F / L_H` (checked against the numbers computed by hand from
`phl`); the estimator path (gaussian k=4, a seed) is deterministic, equals the manual
chain on one frame, sits within 4 s.e. of the Cartesian value on the fixture, does not
change with the chunking, and its HVP machinery matches the matrix applied to `v`;
`replay_ratio` reports frames per Hessian frame -- 0.17 for the campaign's 5,000 on
~29,000 labelled frames, against PFT's 4 -- and the arithmetic is a ratio of dataset
sizes, not of steps.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tests/, where _testlib lives
from _testlib import openqha_src                                # noqa: E402

from openqha.data import dataset                                 # noqa: E402
from openqha.qm_interfaces import orca                           # noqa: E402
from openqha.store import layout, property as prop               # noqa: E402
from openqha_hessian import hvp, phl                             # noqa: E402
from openqha_hessian import smoke_fit                            # noqa: E402
from openqha_hessian.judge import hessian_at                     # noqa: E402

FIX = openqha_src() / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


class ToyHessian(torch.nn.Module):
    """E(x) = 1/2 x^T A x (A = sym(H_t) per frame): forces = -A x, H = A exactly -- the
    curvature the fake's `get_hessian` answers with, so the autograd HVP path can run
    hermetically and the estimator is compared against ITS OWN matrix."""

    def __init__(self, hessian):
        super().__init__()
        a = np.asarray(hessian, dtype=float)
        self.n3 = a.shape[0]
        self.register_buffer("A", torch.as_tensor(0.5 * (a + a.T), dtype=torch.float64))

    def forward(self, data, training=False, compute_force=False, compute_stress=False):
        x = data["positions"]
        x.requires_grad_(True)                       # the leaf, as mace's forward sets it
        n_atoms = self.n3 // 3                       # every frame is the fixture molecule
        reps = x.shape[0] // n_atoms
        a_big = torch.block_diag(*([self.A] * reps))
        flat = x.reshape(-1)
        energy = 0.5 * (flat @ (a_big @ flat))
        return {"forces": -torch.autograd.grad(energy, x, create_graph=True)[0]}


class FakeCalc:
    """Stored numbers answer E / F / `get_hessian`; the HVP protocol (`models[0]`,
    `_atoms_to_batch`, `_clone_batch`) rides on a quadratic toy with the SAME curvature
    sym(H_t) -- what the estimator path needs to run with no engine."""

    def __init__(self, hessian, energy=0.0, forces=None):
        self.h = np.asarray(hessian, dtype=float)
        self.e = float(energy)
        self.f = forces
        self.models = [ToyHessian(self.h)]

    def get_hessian(self, atoms=None):
        n = self.h.shape[0] // 3
        return self.h.reshape(3 * n, n, 3)

    def get_potential_energy(self, atoms=None, **kw):
        return self.e

    def get_forces(self, atoms=None):
        n = len(atoms) if atoms is not None else self.h.shape[0] // 3
        return np.zeros((n, 3)) if self.f is None else np.asarray(self.f)

    def _atoms_to_batch(self, atoms):
        from mace.tools import torch_geometric
        pos = torch.as_tensor(np.asarray(atoms.get_positions()), dtype=torch.float64)
        data = torch_geometric.Data(x=torch.zeros(len(atoms), 1, dtype=torch.float64), positions=pos)
        return torch_geometric.Batch.from_data_list([data])

    def _clone_batch(self, batch):
        return batch.clone()


def main():
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.data import atomic_masses, atomic_numbers
    from ase.io import read

    parsed = orca.parse_hess(layout.orca_level_file(FIX, LEVEL, 0, ".hess"))
    symbols = list(parsed["symbols"])
    x_r = np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
    H_r = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
    H_t = np.load(FIX / "mace" / "basin00" / "hessian_at_{}.npy".format(LEVEL))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src"
        src.mkdir()
        rows = []
        for i in range(10):
            at = Atoms(symbols=symbols, positions=x_r + 1e-4 * i)
            at.calc = SinglePointCalculator(at, energy=-1.0 * i, forces=np.zeros((len(at), 3)))
            at.info = dict(qm9_index="dsgdb9nsd_000035", generator="basin", basin=0, k=i, hessian=H_r)
            rows.append((at, "test"))
        dataset._write_split(src / "test.{}.extxyz".format(LEVEL), rows, reference=True)

        out_dir, info = smoke_fit.build_fit_dataset(src, LEVEL, td / "fit", "fit", seed=0, valid_fraction=0.2)
        check("the fit set holds every labelled frame and marks itself PURPOSE = fit",
              info["PURPOSE"] == "fit" and info["N_FRAMES"] == 10 and info["N_HESSIAN_FRAMES"] == 10, info)
        check("... split by frame, 20 % valid, and both files written",
              (info["N_TRAIN"], info["N_VALID"]) == (8, 2)
              and (out_dir / "train.{}.extxyz".format(LEVEL)).is_file()
              and (out_dir / "valid.{}.extxyz".format(LEVEL)).is_file(), info)
        merged = dataset.merged_file(out_dir, "fit", LEVEL)
        check("... and the merged MACE-form file has all 10 with their split key",
              merged.is_file() and len(read(str(merged), index=":", format="extxyz")) == 10)
        rec = prop.load(out_dir / "fit_dataset.toml")
        check("the fit Record says PURPOSE and names its source",
              rec["Calculation_Info"]["PURPOSE"] == "fit" and str(src) in rec["Calculation_Info"]["SOURCE"])
        same = smoke_fit.build_fit_dataset(src, LEVEL, td / "fit2", "fit", seed=0, valid_fraction=0.2)[1]
        check("the same seed gives the same split", (same["N_TRAIN"], same["N_VALID"]) == (8, 2))
        try:
            smoke_fit.build_fit_dataset(td / "empty", LEVEL, td / "fit3", "fit")
            check("a source with no labelled frame is refused", False)
        except FileNotFoundError as exc:
            check("a source with no labelled frame is refused", "no labelled frame" in str(exc))

        # --- the balance -----------------------------------------------------------------------
        calc = FakeCalc(H_t, energy=-1.0, forces=np.full((len(symbols), 3), 0.01))   # non-zero, or L_F = 0 and there is nothing to balance against
        b_none = smoke_fit.epoch_zero_balance(calc, out_dir / "train.{}.extxyz".format(LEVEL),
                                              forces_weight=100.0)
        exact_none = phl.loss_full(H_t, H_r)
        check("L_H = phl.loss_full, the full matrix (1e-10; the file's 8-decimal positions)",
              abs(b_none["L_H"] - exact_none) < 1e-10, (b_none["L_H"], exact_none))
        check("w_H balances the force term: w_H L_H = w_F L_F (1e-9 relative)",
              abs(b_none["HESSIAN_WEIGHT_BALANCED"] * b_none["L_H"] / b_none["WF_LF"] - 1) < 1e-9, b_none)
        try:
            smoke_fit.epoch_zero_balance(calc, out_dir / "train.{}.extxyz".format(LEVEL),
                                         mode_weighting="entropy")
            check("epoch_zero_balance takes no `mode_weighting` (there is one target)", False)
        except TypeError:
            check("epoch_zero_balance takes no `mode_weighting` (there is one target)", True)
        check("the balance counts the frames it read", b_none["N_FRAMES"] == 8 and b_none["N_HESSIAN_FRAMES"] == 8)

        # --- the estimator path (the balance under the run's probe setting) --------------------
        train_file = out_dir / "train.{}.extxyz".format(LEVEL)
        est1 = smoke_fit.epoch_zero_balance(calc, train_file, forces_weight=100.0,
                                            probe="gaussian", n_probes=4, seed=0, chunk=8)
        est2 = smoke_fit.epoch_zero_balance(calc, train_file, forces_weight=100.0,
                                            probe="gaussian", n_probes=4, seed=0, chunk=8)
        check("the estimator is deterministic (the same seed twice -> bit-identical; k={}, w_H {:.6g})".format(
              est1["N_PROBES"], est1["HESSIAN_WEIGHT_BALANCED"]),
              est1["L_H"] == est2["L_H"] and est1["N_PROBES"] == 4 and est1["PROBE"] == "gaussian")
        check("the exact path records N_PROBES = 0", b_none["N_PROBES"] == 0)

        # the estimate sits within 4 x the closed-form standard error of the mean over the
        # 8 frames (t_train_engine's construction, gaussian k=4; the s.e. is the yardstick,
        # never a guessed tolerance)
        exact_frames, var_sum = [], 0.0
        for atoms in read(str(train_file), index=":", format="extxyz"):
            n3 = 3 * len(atoms)
            h_r = np.asarray(atoms.info.get("REF_hessian", atoms.info.get("hessian")), dtype=float).reshape(n3, n3)
            h_e = hessian_at(calc, atoms)
            exact_frames.append(phl.loss_full(h_e, h_r))
            var_sum += phl.estimator_variance(h_e, h_r, k=4)["gaussian"]
        exact_mean = float(np.mean(exact_frames))
        se_mean = float(np.sqrt(var_sum)) / len(exact_frames)
        z = abs(est1["L_H"] - exact_mean) / se_mean
        check("the estimate = the Cartesian value within 4 s.e. (z = {:.2f}; est {:.6e} vs exact {:.6e}, s.e. {:.1e})".format(
              z, est1["L_H"], exact_mean, se_mean),
              z <= 4.0 and abs(exact_mean - b_none["L_H"]) <= 1e-12 * b_none["L_H"])

        # the wiring identity: the chunked HVP machinery = the exact matrix applied to v,
        # and a chunk = the single-frame call (same per-frame draws, block-diagonal batch)
        one_file = td / "one.{}.extxyz".format(LEVEL)
        dataset._write_split(one_file, [rows[0]], reference=True)
        atoms0 = read(str(one_file), index="0")
        n0 = len(atoms0)
        rng_w = np.random.default_rng(11)
        v_a = rng_w.standard_normal((4, 3 * n0))
        v_b = rng_w.standard_normal((4, 3 * n0))          # different probes: a frame swap must show
        hv_batch = hvp.hvp_from_atoms_batch(calc, [atoms0, atoms0], [v_a, v_b])
        hv_a = hvp.hvp_from_atoms(calc, atoms0, v_a)
        hv_b = hvp.hvp_from_atoms(calc, atoms0, v_b)
        h_mat = hessian_at(calc, atoms0)
        ref_a = (v_a @ h_mat).reshape(4, n0, 3)
        ref_b = (v_b @ h_mat).reshape(4, n0, 3)
        rel_wire = max(float(np.abs(hv_batch[0] - ref_a).max() / np.abs(ref_a).max()),
                       float(np.abs(hv_batch[1] - ref_b).max() / np.abs(ref_b).max()))
        rel_chunk = max(float(np.abs(hv_batch[0] - hv_a).max() / np.abs(hv_a).max()),
                        float(np.abs(hv_batch[1] - hv_b).max() / np.abs(hv_b).max()))
        check("H_theta v through the chunked machinery = the matrix applied to v, per frame ({:.1e} rel <= 1e-8)".format(
              rel_wire), rel_wire <= 1e-8)
        check("... and a two-frame chunk = the single-frame calls, per frame ({:.1e} rel <= 1e-12)".format(rel_chunk),
              rel_chunk <= 1e-12)

        # one frame: the function's value IS the manual chain (rng -> make_probes -> HVP -> estimator)
        b_one = smoke_fit.epoch_zero_balance(calc, one_file, forces_weight=100.0,
                                             probe="gaussian", n_probes=4, seed=7)
        h_r0 = np.asarray(atoms0.info.get("REF_hessian", atoms0.info.get("hessian")),
                          dtype=float).reshape(3 * n0, 3 * n0)
        v7, r7, info7 = phl.make_probes(h_r0, mode="gaussian", k=4, rng=np.random.default_rng(7))
        manual = phl.estimator_from_products(hvp.hvp_from_atoms(calc, atoms0, v7), r7, info7)
        check("one frame: the value is the manual chain to 1e-12 ({:.6e} vs {:.6e})".format(b_one["L_H"], manual),
              abs(b_one["L_H"] - manual) <= 1e-12 * abs(manual))

        # chunking is a pure compute reorganization: the same seed, chunk 1 vs 8, one value
        est_c1 = smoke_fit.epoch_zero_balance(calc, train_file, forces_weight=100.0,
                                              probe="gaussian", n_probes=4, seed=0, chunk=1)
        check("chunk=1 vs chunk=8 at the same seed: the same value ({:.1e} rel)".format(
              abs(est_c1["L_H"] - est1["L_H"]) / abs(est1["L_H"])),
              abs(est_c1["L_H"] - est1["L_H"]) <= 1e-10 * abs(est1["L_H"]))

        try:
            smoke_fit.epoch_zero_balance(calc, train_file, probe="modes")
            check("an unknown probe name is refused", False)
        except ValueError as exc:
            check("an unknown probe name is refused", "probe must be" in str(exc))
        est_r = smoke_fit.epoch_zero_balance(calc, train_file, forces_weight=100.0,
                                             probe="rademacher", n_probes=4, seed=0)
        check("rademacher runs too: a finite value, k recorded",
              np.isfinite(est_r["L_H"]) and est_r["N_PROBES"] == 4)

    # --- the replay arithmetic -----------------------------------------------------------------
    r = smoke_fit.replay_ratio(n_train=99000, n_hessian=29000, num_samples_pt=5000)
    check("campaign, 5,000: 0.17 SPICE frames per Hessian frame against PFT's 4",
          abs(r["REPLAY_PER_HESSIAN_FRAME"] - 0.172) < 5e-3 and r["PFT_REFERENCE"] == 4.0, r)
    r20 = smoke_fit.replay_ratio(99000, 29000, 20000)
    check("campaign, 20,000: 0.69 per Hessian frame (4x the frames, 4x the ratio -- it is linear)",
          abs(r20["REPLAY_PER_HESSIAN_FRAME"] - 0.690) < 5e-3
          and abs(r20["REPLAY_PER_HESSIAN_FRAME"] / r["REPLAY_PER_HESSIAN_FRAME"] - 4.0) < 1e-9, r20)
    smoke = smoke_fit.replay_ratio(65, 33, 5000)
    check("the SAME flag means 152 per Hessian frame on the smoke set: a replay setting "
          "quoted without this ratio says nothing",
          abs(smoke["REPLAY_PER_HESSIAN_FRAME"] - 151.5) < 0.5, smoke)
    check("no replay -> 0, and a set with no Label -> None (not a division by zero)",
          smoke_fit.replay_ratio(10, 5, 0)["REPLAY_PER_HESSIAN_FRAME"] == 0.0
          and smoke_fit.replay_ratio(10, 0, 100)["REPLAY_PER_HESSIAN_FRAME"] is None)

    print("\n{} checks, {} failed".format(23, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
