"""The fit set, the epoch-0 balance and the replay
arithmetic -- no engine (a fake calculator answers with stored Hessians).

Asserted: `build_fit_dataset` re-splits labelled frames by frame, marks `PURPOSE = fit`
and writes the merged file; `epoch_zero_balance` returns the three terms of eq. 11 on the
base model and `w_H = w_F L_F / L_H` (checked against the numbers computed by hand from
`phl`), and the entropy and flat weightings give very different `w_H` (the whole reason
the weight is measured rather than quoted); `replay_ratio` reports frames per Hessian
frame -- 0.17 for the campaign's 5,000 on ~29,000 labelled frames, against PFT's 4 -- and
the arithmetic is a ratio of dataset sizes, not of steps.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tests/, where _testlib lives
from _testlib import openqha_src                                # noqa: E402

from openqha.data import dataset                                 # noqa: E402
from openqha.qm_interfaces import orca                           # noqa: E402
from openqha.store import layout, property as prop               # noqa: E402
from openqha_hessian import phl                                  # noqa: E402
from openqha_hessian import smoke_fit                            # noqa: E402

FIX = openqha_src() / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


class FakeCalc:
    def __init__(self, hessian, energy=0.0, forces=None):
        self.h = np.asarray(hessian, dtype=float)
        self.e = float(energy)
        self.f = forces

    def get_hessian(self, atoms=None):
        n = self.h.shape[0] // 3
        return self.h.reshape(3 * n, n, 3)

    def get_potential_energy(self, atoms=None, **kw):
        return self.e

    def get_forces(self, atoms=None):
        n = len(atoms) if atoms is not None else self.h.shape[0] // 3
        return np.zeros((n, 3)) if self.f is None else np.asarray(self.f)


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

    print("\n{} checks, {} failed".format(13, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
