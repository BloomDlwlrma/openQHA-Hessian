"""Ticket 14 of the Hessian-learning set: the judge's arithmetic, its set-aside rule and
its verdict lines -- no engine (a fake calculator answers with stored Hessians).

Asserted: a per-frame row reproduces `hessian_compare`'s numbers and the training target
`||H - H_r||_F^2 / (9 N^2)` exactly (the gate's quantity); `ScaledCalculator(0.9)` gives
frequencies 0.9x to 1e-10 and a Hessian scaled by 0.81; the distribution of a molecule
(in_distribution beats out_of_molecule beats interpolation); class and distribution
aggregation add up; the `[Anharmonic]` rule sets aside a 20 cm^-1 mode, keeps a 40 cm^-1
one, and sets aside a 40 cm^-1 mode whose FD self-check is 7 cm^-1; a verdict line with
nothing to measure is `-` and never a silent PASS; the must-pass and must-fail lines
behave; `forgetting` compares two calculators on a frame file; the Record round-trips.

Ticket 22 (gate rows and reference rows): `rms_bin` at the edges; `aggregate_displacement`
groups by (distribution, bin) with H over the Hessian frames and E/F over all; the
verdict carries GATE yes / no, `verdict_of` reads gate rows only (worsening a reference
row -- the low-mode line, the held-out Hessian line, the ramp -- changes nothing;
worsening a gate row fails it); `frame_rows` gives an E/F-only frame a row with `-` H
metrics; `ramp_schedule`; `ramp_one` on a fake calculator that explodes after a chosen
number of force calls fails at the chosen temperature and survives otherwise.
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
from openqha.thermochem import hessian as hessian_mod            # noqa: E402
from openqha.thermochem import hessian_compare as hc             # noqa: E402
from openqha_hessian import judge                                # noqa: E402
from openqha_hessian import phl                                  # noqa: E402

FIX = openqha_src() / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


class FakeCalc:
    """Answers with a stored Hessian (and, for the forgetting line, a scaled energy)."""

    def __init__(self, hessian, energy=0.0, forces=None):
        self.h = np.asarray(hessian, dtype=float)
        self.e = float(energy)
        self.f = forces
        self.r_max = 5.0

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

    parsed = orca.parse_hess(layout.orca_level_file(FIX, LEVEL, 0, ".hess"))
    symbols = list(parsed["symbols"])
    x_r = np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
    H_r = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
    H_t = np.load(FIX / "mace" / "basin00" / "hessian_at_{}.npy".format(LEVEL))
    atoms = Atoms(symbols=symbols, positions=x_r)

    # --- ScaledCalculator ---------------------------------------------------------------------
    base = FakeCalc(H_t)
    scaled = judge.ScaledCalculator(base, 0.9)
    h_s = judge.hessian_at(scaled, atoms)
    check("ScaledCalculator: the Hessian is scaled by s^2", np.abs(h_s - 0.81 * H_t).max() < 1e-12)
    f_base = hc.projected(H_t, masses, x_r)["freq"]
    f_scaled = hc.projected(h_s, masses, x_r)["freq"]
    check("... so every frequency is 0.9x (1e-10)", np.abs(f_scaled - 0.9 * f_base).max() < 1e-10,
          np.abs(f_scaled - 0.9 * f_base).max())
    check("... and the forces are scaled by s",
          np.abs(scaled.get_forces(atoms) - 0.9 * base.get_forces(atoms)).max() < 1e-12)

    # --- one frame row -----------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        at = atoms.copy()
        at.calc = SinglePointCalculator(at, energy=-1.0, forces=np.zeros((len(at), 3)))
        at.info = dict(qm9_index="dsgdb9nsd_000018", generator="basin", basin=0, k=0, hessian=H_r)
        dataset._write_split(td / "test.{}.extxyz".format(LEVEL), [(at, "test")], reference=True)
        rows, anh = judge.frame_rows(td, "ds", LEVEL, FakeCalc(H_t), base_calc=FakeCalc(H_t),
                                     splits=("test",), index=[dict(qm9_index="dsgdb9nsd_000018",
                                                                   classes="aldehyde;ketone")])
        cmp_e = hc.compare_hessians(H_t, H_r, masses, x_r)
        r = rows[0]
        # the tolerances are what an extxyz round trip allows: positions are written to
        # 8 decimals (5e-9 A here), which moves the Eckart projector and with it every
        # projected quantity -- 2e-10 cm^-1 on the low-mode MAE, 3e-14 on the loss
        check("a frame row reproduces hessian_compare's numbers",
              len(rows) == 1 and abs(r["freq_mae_low_cm"] - cmp_e["FREQ_MAE_LOW_CM"]) < 1e-8
              and abs(r["freq_mae_cm"] - cmp_e["FREQ_MAE_CM"]) < 1e-8
              and abs(r["hessian_mae"] - cmp_e["HESSIAN_MAE"]) < 1e-12
              and abs(r["mixing"] - cmp_e["MIXING"]) < 1e-10, r)
        check("the base columns are filled and the classes come from index.dat",
              r["base_freq_mae_cm"] == r["freq_mae_cm"] and r["classes"] == "aldehyde;ketone")
        check("a shipped molecule is in_distribution, not out_of_molecule",
              r["distribution"] == "in_distribution")

    check("a pinned, non-shipped molecule is out_of_molecule",
          judge.distribution_of("dsgdb9nsd_000044") == "out_of_molecule")
    check("any other molecule is interpolation", judge.distribution_of("dsgdb9nsd_099999") == "interpolation")

    # --- the anharmonic rule --------------------------------------------------------------------------
    anh = judge.anharmonic_modes([20.0, 40.0, 200.0], "m", 0)
    check("a 20 cm^-1 mode is set aside, a 40 cm^-1 one is not",
          len(anh) == 1 and anh[0]["MODE"] == 0 and "omega_r" in anh[0]["REASON"], anh)
    anh = judge.anharmonic_modes([40.0, 200.0], "m", 0, profiles={0: 7.0, 1: 1.0})
    check("a 40 cm^-1 mode whose FD self-check is 7 cm^-1 is set aside too",
          len(anh) == 1 and anh[0]["MODE"] == 0 and "FD self-check" in anh[0]["REASON"], anh)

    # --- aggregation ------------------------------------------------------------------------------------
    rows = [dict(qm9_index="a", distribution="interpolation", classes="epoxide", freq_mae_low_cm=2.0,
                 freq_mae_cm=4.0, hessian_mae=0.1, eigval_mae_eckart=0.2, loss_cartesian=0.5,
                 base_freq_mae_low_cm=4.0, base_freq_mae_cm=8.0, base_hessian_mae=0.2,
                 base_eigval_mae_eckart=0.4, base_loss_cartesian=1.0),
            dict(qm9_index="b", distribution="interpolation", classes="epoxide;amide", freq_mae_low_cm=4.0,
                 freq_mae_cm=6.0, hessian_mae=0.3, eigval_mae_eckart=0.4, loss_cartesian=1.5,
                 base_freq_mae_low_cm=8.0, base_freq_mae_cm=12.0, base_hessian_mae=0.6,
                 base_eigval_mae_eckart=0.8, base_loss_cartesian=3.0),
            dict(qm9_index="c", distribution="in_distribution", classes="-", freq_mae_low_cm=1.0,
                 freq_mae_cm=1.0, hessian_mae=0.05, eigval_mae_eckart=0.05, loss_cartesian=0.25,
                 base_freq_mae_low_cm=1.0, base_freq_mae_cm=1.0, base_hessian_mae=0.05,
                 base_eigval_mae_eckart=0.05, base_loss_cartesian=0.25)]
    dist, cls = judge.aggregate(rows)
    check("per distribution: the means and the molecule counts",
          [d["DISTRIBUTION"] for d in dist] == ["interpolation", "in_distribution"]
          and dist[0]["N_FRAMES"] == 2 and dist[0]["N_MOLECULES"] == 2
          and dist[0]["FREQ_MAE_LOW_CM"] == 3.0 and dist[0]["BASE_FREQ_MAE_LOW_CM"] == 6.0, dist)
    check("per class: a molecule counts for every class it is in",
          {c["CLASS"]: c["N_FRAMES"] for c in cls} == {"epoxide": 2, "amide": 1}
          and [c for c in cls if c["CLASS"] == "amide"][0]["FREQ_MAE_LOW_CM"] == 4.0, cls)

    # --- the verdict --------------------------------------------------------------------------------------
    lines = judge.verdict(dist, [], None, noise_floor_cm=10.0)
    by = {l["LINE"]: l for l in lines}
    check("the low-mode line passes at 3.0 cm^-1 and names the Label's grid noise",
          by["held_out_low_mode_mae_cm"]["RESULT"] == "PASS" and "grid noise" in by["held_out_low_mode_mae_cm"]["NOTE"])
    check("no msRRHO Record and no SPICE draw -> '-', never a silent PASS",
          by["model_error_s_ref_cal_per_mol_K"]["RESULT"] == "-" and by["forgetting"]["RESULT"] == "-")
    check("the in-distribution line reads the worst HIP metric against the base",
          abs(by["in_distribution_degradation"]["VALUE"]) < 1e-12
          and by["in_distribution_degradation"]["RESULT"] == "PASS", by.get("in_distribution_degradation"))
    bad = [dict(d) for d in dist]
    bad[0]["FREQ_MAE_LOW_CM"] = 30.0
    check("the low-mode line reads FAIL at 30 cm^-1",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(bad, [], None)}["held_out_low_mode_mae_cm"] == "FAIL")
    worse = [dict(d) for d in dist]
    worse[1]["FREQ_MAE_CM"] = 2.0                                 # twice the base's on the shipped molecules
    check("the no-degradation line fails when an in-distribution metric doubles",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(worse, [], None)}["in_distribution_degradation"] == "FAIL")
    thermo = [dict(QM9_INDEX="a", SOURCE="x", S_MSRRHO=70.0, MODEL_ERROR_S_REF=-0.35, N_ANHARMONIC=0)]
    check("the entropy line reads FAIL at 0.35 cal/mol/K",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(dist, thermo, None)}["model_error_s_ref_cal_per_mol_K"] == "FAIL")
    forget = dict(N_FRAMES=10, E_RATIO=1.05, F_RATIO=1.30)
    check("the forgetting line fails at a 30 % force ratio",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(dist, [], forget)}["forgetting"] == "FAIL")

    # --- forgetting on two calculators ---------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        from ase.io import write
        frames = []
        for i in range(3):
            a = Atoms("H2O", positions=np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]) + 0.01 * i)
            a.info["REF_energy"] = 0.0
            a.arrays["REF_forces"] = np.zeros((3, 3))
            frames.append(a)
        write(str(Path(td) / "spice.xyz"), frames, format="extxyz")
        f = judge.forgetting(FakeCalc(np.eye(9), energy=0.03), FakeCalc(np.eye(9), energy=0.003),
                             Path(td) / "spice.xyz")
        check("forgetting: per-atom E RMSE and the ratio to the base",
              f["N_FRAMES"] == 3 and abs(f["ENGINE_E_RMSE_MEV_PER_ATOM"] - 10.0) < 1e-9
              and abs(f["E_RATIO"] - 10.0) < 1e-9, f)
        check("a missing SPICE file gives None, not an exception",
              judge.forgetting(FakeCalc(np.eye(9)), FakeCalc(np.eye(9)), Path(td) / "nope.xyz") is None)

    # --- the Record ---------------------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        info = {k: (0.0 if t[0] == "Double" else 0 if t[0] == "Integer" else [] if t[0].startswith("ArrayOf") else "x")
                for k, t in judge.SCHEMA["Calculation_Info"].items()}
        info.update(RUN="r", VERDICT="PASS")
        out = dict(info=info, frames=rows, distributions=dist, classes=cls,
                   thermochemistry=thermo, anharmonic=anh, forgetting=None, verdict=lines,
                   run_dir=Path(td))
        judge.write_record(out)
        rec = prop.load(Path(td) / "judge.toml")
        check("judge.toml round-trips with the five blocks and NORMAL TERMINATION",
              rec["Calculation_Status"]["STATUS"] == prop.NORMAL_TERMINATION
              and len(rec["Distribution"]) == 2 and len(rec["Class"]) == 2 and len(rec["Verdict"]) == len(lines),
              sorted(rec))
        check("the Record carries the package identity: the schema holds the two keys and the round-trip keeps them",
              {"HL_PACKAGE_VERSION", "HL_PACKAGE_COMMIT"} <= set(judge.SCHEMA["Calculation_Info"])
              and "HL_PACKAGE_VERSION" in rec["Calculation_Info"] and "HL_PACKAGE_COMMIT" in rec["Calculation_Info"])
        check("judge.out and judge.dat are written",
              (Path(td) / "judge.out").is_file() and (Path(td) / "judge.dat").is_file())

    # ================================================================== ticket 22
    check("rms_bin: 0 -> '0', 0.05 -> '<0.08', 0.08 -> '<0.15', 0.149 -> '<0.15', 0.15 -> '>=0.15'",
          [judge.rms_bin(x) for x in (0.0, 0.05, 0.08, 0.149, 0.15, 1.0)] == ["0", "<0.08", "<0.15", "<0.15", ">=0.15", ">=0.15"])
    disp_in = [dict(qm9_index="a", distribution="interpolation", rms_displacement_A=0.0, rms_bin="0", has_hessian=True,
                    hessian_mae=0.1, freq_mae_cm=4.0, freq_mae_low_cm=2.0, e_err_mev_per_atom=1.0, f_rmse_mev_a=10.0,
                    base_hessian_mae=0.2, base_freq_mae_cm=8.0, base_freq_mae_low_cm=4.0, base_e_err_mev_per_atom=2.0,
                    base_f_rmse_mev_a=20.0),
               dict(qm9_index="a", distribution="interpolation", rms_displacement_A=0.05, rms_bin="<0.08", has_hessian=False,
                    hessian_mae=None, freq_mae_cm=None, freq_mae_low_cm=None, e_err_mev_per_atom=3.0, f_rmse_mev_a=30.0,
                    base_e_err_mev_per_atom=6.0, base_f_rmse_mev_a=60.0),
               dict(qm9_index="b", distribution="interpolation", rms_displacement_A=0.06, rms_bin="<0.08", has_hessian=True,
                    hessian_mae=0.3, freq_mae_cm=6.0, freq_mae_low_cm=3.0, e_err_mev_per_atom=5.0, f_rmse_mev_a=50.0,
                    base_hessian_mae=0.2, base_freq_mae_cm=5.0, base_freq_mae_low_cm=2.5, base_e_err_mev_per_atom=4.0,
                    base_f_rmse_mev_a=40.0)]
    dr = judge.aggregate_displacement(disp_in)
    check("aggregate_displacement: bins in order, H over the Hessian frames only (1 of 2 in '<0.08'), E/F over all, GATE no",
          [(d["RMS_BIN"], d["N_FRAMES"], d["N_HESSIAN"]) for d in dr] == [("0", 1, 1), ("<0.08", 2, 1)]
          and dr[1]["HESSIAN_MAE"] == 0.3 and dr[1]["E_MAE_MEV_PER_ATOM"] == 4.0 and dr[1]["F_RMSE_MEV_A"] == 40.0
          and dr[1]["N_MOLECULES"] == 2 and all(d["GATE"] == "no" for d in dr), dr)
    check("aggregate ignores the E/F-only frames (2 Hessian frames in the distribution row)",
          judge.aggregate(disp_in)[0][0]["N_FRAMES"] == 2)
    # the gate / reference split
    ramp_rows = [dict(QM9_INDEX="a", WHICH="engine", SURVIVED=False, FAIL_T_K=150.0, FAIL_PS=1.0, MAX_RATIO=1.6, MIN_RATIO=0.9, N_STEPS=10, SECONDS=1.0),
                 dict(QM9_INDEX="a", WHICH="base", SURVIVED=False, FAIL_T_K=300.0, FAIL_PS=2.0, MAX_RATIO=1.6, MIN_RATIO=0.9, N_STEPS=20, SECONDS=1.0)]
    lines = judge.verdict(dist, thermo, forget, noise_floor_cm=10.0, ramp_rows=ramp_rows, disp_rows=dr)
    by = {l["LINE"]: l for l in lines}
    check("the verdict's rows carry GATE (S0-C-59): the Hessian matrix / in_distribution / forgetting yes; low-mode / entropy / held-out bins / ramp no",
          {l["LINE"]: l["GATE"] for l in lines} == {"held_out_hessian_cartesian": "yes", "in_distribution_degradation": "yes",
                                                     "forgetting": "yes", "held_out_low_mode_mae_cm": "no",
                                                     "model_error_s_ref_cal_per_mol_K": "no",
                                                     "held_out_generator_hessian_vs_base": "no", "md_ramp_K": "no"},
          {l["LINE"]: l["GATE"] for l in lines})
    check("the Hessian gate: engine 1.0 against base 2.0 on the interpolation row -> ratio - 1 = -0.5, PASS (an improvement)",
          abs(by["held_out_hessian_cartesian"]["VALUE"] + 0.5) < 1e-12 and by["held_out_hessian_cartesian"]["RESULT"] == "PASS",
          by["held_out_hessian_cartesian"])
    worse_h = [dict(d) for d in dist]
    worse_h[0]["LOSS_CARTESIAN"] = 2.5                              # 25 % worse than the base's 2.0
    ok_forget_probe = dict(N_FRAMES=10, E_RATIO=1.0, F_RATIO=1.0)
    check("... engine 2.5 against base 2.0 fails the Hessian gate and the verdict",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(worse_h, [], ok_forget_probe)}["held_out_hessian_cartesian"] == "FAIL"
          and judge.verdict_of(judge.verdict(worse_h, [], ok_forget_probe)) == "FAIL")
    check("the reference rows read: held-out H worse than the base (+50 %, FAIL), the ramp failing at 150 K against the base's 300 (FAIL)",
          abs(by["held_out_generator_hessian_vs_base"]["VALUE"] - 0.5) < 1e-12 and by["held_out_generator_hessian_vs_base"]["RESULT"] == "FAIL"
          and by["md_ramp_K"]["VALUE"] == 150.0 and by["md_ramp_K"]["THRESHOLD"] == 300.0 and by["md_ramp_K"]["RESULT"] == "FAIL")
    check("... and NONE of them moves the verdict, which the gate rows set (forgetting FAIL here -> FAIL; with a passing forgetting -> PASS)",
          judge.verdict_of(lines) == "FAIL"
          and judge.verdict_of(judge.verdict(dist, [], dict(N_FRAMES=10, E_RATIO=1.0, F_RATIO=1.0),
                                             ramp_rows=ramp_rows, disp_rows=dr)) == "PASS")
    bad_low = [dict(d) for d in dist]
    bad_low[0]["FREQ_MAE_LOW_CM"] = 30.0
    ok = dict(N_FRAMES=10, E_RATIO=1.0, F_RATIO=1.0)
    check("worsening the low-mode REFERENCE row to 30 cm^-1 reads FAIL and leaves the verdict PASS (a quantity computed from the matrix)",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(bad_low, [], ok)}["held_out_low_mode_mae_cm"] == "FAIL"
          and judge.verdict_of(judge.verdict(bad_low, [], ok)) == "PASS")
    check("worsening the entropy REFERENCE row to 0.35 cal/mol/K reads FAIL and leaves the verdict PASS",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(dist, thermo, ok)}["model_error_s_ref_cal_per_mol_K"] == "FAIL"
          and judge.verdict_of(judge.verdict(dist, thermo, ok)) == "PASS")
    worse_ref = [dict(d) for d in dr]
    for d in worse_ref:
        if d["RMS_BIN"] != "0":
            d["HESSIAN_MAE"] = 10.0 * (d["BASE_HESSIAN_MAE"] or 1.0)
    check("worsening a REFERENCE row (the held-out H bins 10x the base) reads FAIL and leaves the verdict PASS",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(dist, [], ok, disp_rows=worse_ref)}["held_out_generator_hessian_vs_base"] == "FAIL"
          and judge.verdict_of(judge.verdict(dist, [], ok, disp_rows=worse_ref)) == "PASS")
    check("worsening a GATE row (the forgetting ratio 1.30) fails the verdict",
          judge.verdict_of(judge.verdict(dist, [], forget)) == "FAIL")
    check("ramp_schedule: 5, 10, ..., 600 has 120 stages; to 20 K four",
          len(judge.ramp_schedule()) == 120 and judge.ramp_schedule(max_K=20) == [5.0, 10.0, 15.0, 20.0])

    # --- E/F-only frames get a row ---------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        at = atoms.copy()
        at.calc = SinglePointCalculator(at, energy=-1.0, forces=np.zeros((len(at), 3)))
        at.info = dict(qm9_index="dsgdb9nsd_000044", generator="displaced", basin=0, k=3, rms_displacement_A=0.09)
        dataset._write_split(td / "test.{}.extxyz".format(LEVEL), [(at, "test")], reference=True)
        rows_ef, _anh = judge.frame_rows(td, "ds", LEVEL, FakeCalc(H_t, energy=-1.0 + 0.001 * len(at)), base_calc=FakeCalc(H_t, energy=-1.0),
                                         splits=("test",), index=[dict(qm9_index="dsgdb9nsd_000044", generator="displaced", basin=0, k=3,
                                                                       classes="epoxide", held_out_generator="yes")])
        r = rows_ef[0]
        check("an E/F-only displaced frame has a row: no H metrics, E error 1 meV/atom, bin '<0.15', held out, out_of_molecule",
              len(rows_ef) == 1 and r["has_hessian"] is False and r["hessian_mae"] is None and r["freq_mae_cm"] is None
              and abs(r["e_err_mev_per_atom"] - 1.0) < 1e-9 and r["base_e_err_mev_per_atom"] == 0.0
              and r["rms_bin"] == "<0.15" and r["held_out_generator"] == "yes" and r["distribution"] == "out_of_molecule", r)

    # --- the ramp on a fake calculator that explodes after a chosen number of force calls -----------------
    from ase.calculators.calculator import Calculator, all_changes

    class Explodes(Calculator):
        """Harmonic to the start geometry for `fail_after` force calls, then a strong push
        outwards: the pair distances blow up within a window and the ramp must catch it."""
        implemented_properties = ["energy", "forces"]

        def __init__(self, x0, fail_after, k=5.0):
            super().__init__()
            self.x0, self.fail_after, self.k, self.calls = np.asarray(x0, float), int(fail_after), float(k), 0

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            d = atoms.get_positions() - self.x0
            if self.calls > self.fail_after:
                com = atoms.get_positions().mean(axis=0)
                f = 50.0 * (atoms.get_positions() - com)
            else:
                f = -self.k * d
            self.results = dict(energy=float(0.5 * self.k * (d * d).sum()), forces=f)

    h2o = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    ramp_kw = dict(max_K=20.0, step_K=5.0, step_ps=0.01, timestep_fs=1.0, window_steps=5, optimise=False, seed=1)
    survived = judge.ramp_one(Explodes(h2o.positions, 10 ** 6), h2o, **ramp_kw)
    check("ramp_one: a harmonic calculator survives to the 20 K ceiling in 4 x 10 steps",
          survived["SURVIVED"] and survived["FAIL_T_K"] == 20.0 and survived["N_STEPS"] == 40 and abs(survived["FAIL_PS"] - 0.04) < 1e-12
          and 0.9 < survived["MIN_RATIO"] <= 1.0 <= survived["MAX_RATIO"] < 1.1, survived)
    boom = judge.ramp_one(Explodes(h2o.positions, 15), h2o, **ramp_kw)
    check("... one that explodes after 15 force calls fails at the second stage (10 K), within the third window, ratio > 1.5",
          not boom["SURVIVED"] and boom["FAIL_T_K"] == 10.0 and boom["N_STEPS"] < 40 and boom["MAX_RATIO"] > 1.5
          and 0.01 < boom["FAIL_PS"] <= boom["N_STEPS"] / 1000.0, boom)
    rr = judge.md_ramp(Explodes(h2o.positions, 15), Explodes(h2o.positions, 10 ** 6), ["m"], {"m": h2o}, **ramp_kw)
    check("md_ramp: one row per (molecule, which); the engine fails at 10 K, the base survives -> the reference line reads FAIL, the verdict ignores it",
          [(r["QM9_INDEX"], r["WHICH"], r["SURVIVED"]) for r in rr] == [("m", "engine", False), ("m", "base", True)]
          and {l["LINE"]: l["RESULT"] for l in judge.verdict(dist, [], ok, ramp_rows=rr)}["md_ramp_K"] == "FAIL"
          and judge.verdict_of(judge.verdict(dist, [], ok, ramp_rows=rr)) == "PASS", rr)

    check("the gate closed (S0-C-60): verdict_of(gate=False) is REPORTED whatever the rows say; open, the same rows read FAIL",
          judge.verdict_of(judge.verdict(worse_h, [], ok_forget_probe), gate=False) == "REPORTED"
          and judge.verdict_of(judge.verdict(worse_h, [], ok_forget_probe), gate=True) == "FAIL" and judge.GATE_CLOSED == "REPORTED")

    print("\n{} checks, {} failed".format(40, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
