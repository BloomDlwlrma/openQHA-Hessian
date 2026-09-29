"""Ticket 14 of the Hessian-learning set: the judge on the REAL engine, with its two
calibrations.

INTEGRATION (engine, CPU). Builds a one-molecule Dataset from the 2-methyloxirane frame
fixture (basin frame with a reference Hessian) and judges MACE-OFF23_medium against
itself:

  * must-PASS: the base model's numbers reproduce `hessian_compare`'s known ones and
    the training target `||H - H_r||_F^2/(9N^2)` (the gate's quantity); the
    engine columns equal the base columns (it IS the base); against `H_r := H_base` the
    loss is ~0; no degradation line FAILs.
  * must-FAIL: `ScaledCalculator(0.9)` -- every frequency 0.9x -- fails the low-mode line.
  * and the base model ITSELF fails the low-mode line on this molecule (12.7 cm^-1
    against the 8.5 threshold): 2-methyloxirane is an out-of-distribution ring, which is
    where MACE-OFF23's curvature error lives (S0-C-41). A judge that passed it here would
    not be measuring what the fine-tune is for.

Ticket 22 (2026-09-22, S0-C-58/59): the gate is the Hessian MATRIX against the Label,
engine vs base (||dH||^2/(9N^2), the training target); the low-mode line and the entropy
are computed from it and are reference rows, like the RMS bins and the MD ramp; the 0.9x
potential (Hessian x0.81) FAILs the Hessian gate and the verdict, the base against itself
reads exactly 0 and PASSes; the fixture's four displaced frames
(E/F labels, no Hessian; rms 0.058-0.11 A) are held-out frames that populate the
displacement bins with E/F errors on the base model; a real MD ramp with `max_K 20`,
5 K steps of 0.05 ps on the molecule survives on the base (the `[Ramp]` rows, the
reference line '-' without a base... engine == base here, so PASS); the Record holds
`[[Displacement]]` and `[[Ramp]]`.

SKIPs without the model.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tests/, where _testlib lives
from _testlib import openqha_src                                # noqa: E402

FIX = openqha_src() / "tests" / "data" / "methyloxirane_frames"
LEVEL = "wb97m-d3bj_def2-tzvppd"
MACE_LEVEL = "mace-off23_medium"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    from openqha.data import dataset, frames
    from openqha.potentials import engine
    from openqha.store import property as prop
    from openqha_hessian import judge
    from openqha_hessian import phl
    try:
        calc, name, prov = engine.calculator(device="cpu")
    except Exception as exc:                                     # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0
    print("  engine {}  weights {}".format(name, prov["weights_path"]))

    ref = frames.read_frames(FIX / "basin.{}.extxyz".format(LEVEL))
    mace_basin = frames.read_frames(FIX / "basin.{}.extxyz".format(MACE_LEVEL))[0]
    with tempfile.TemporaryDirectory() as tmp:
        from openqha.data import dataset as dataset_mod
        d = Path(dataset_mod.datasets_dir(Path(tmp), "smoke", "smoke"))
        d.mkdir(parents=True)
        disp = frames.read_frames(FIX / "displaced.{}.extxyz".format(LEVEL))
        for a in disp:                                             # held-out frames: E/F labels only (S0-C-54)
            a.info.pop("hessian", None)
        dataset._write_split(d / "test.{}.extxyz".format(LEVEL), [(a, "test") for a in ref] + [(a, "test") for a in disp],
                             reference=True)

        rows, anh = judge.frame_rows(d, "smoke", LEVEL, calc, base_calc=calc, splits=("test",),
                                     index=[dict(qm9_index="dsgdb9nsd_000044", classes="epoxide;small_ring")])
        r = rows[0]
        check("five rows: the basin frame with a Hessian and four displaced E/F-only frames, bins '<0.08' / '<0.15'",
              len(rows) == 5 and rows[0]["has_hessian"] and all(not x["has_hessian"] for x in rows[1:])
              and all(x["hessian_mae"] is None and x["e_err_mev_per_atom"] is not None and x["f_rmse_mev_a"] is not None
                      for x in rows[1:])
              and {x["rms_bin"] for x in rows[1:]} <= {"<0.08", "<0.15"} and rows[0]["rms_bin"] == "0",
              [(x["generator"], x["rms_bin"], x["has_hessian"]) for x in rows])
        cart = phl.loss_full(mace_basin.info["hessian"], ref[0].info["hessian"])
        check("the judge's loss_cartesian on the basin frame = phl.loss_full(H_t, H_r) (1e-6 relative)",
              abs(r["loss_cartesian"] / cart - 1) < 1e-6 and r["loss_cartesian"] > 0,
              (r["loss_cartesian"], cart))
        check("the engine columns equal the base columns (it IS the base model)",
              abs(r["freq_mae_cm"] - r["base_freq_mae_cm"]) < 1e-9
              and abs(r["loss_cartesian"] - r["base_loss_cartesian"]) < 1e-15,
              (r["freq_mae_cm"], r["base_freq_mae_cm"], r["loss_cartesian"], r["base_loss_cartesian"]))
        check("the frame carries its classes, its distribution and the Label's noise floor",
              r["classes"] == "epoxide;small_ring" and r["distribution"] == "out_of_molecule"
              and r["noise_floor_cm"] > 0, r)
        print("    2-methyloxirane basin: low-mode MAE {:.2f}, full MAE {:.2f} cm^-1, "
              "||dH||^2/9N^2 {:.4e}, noise floor {:.1f} cm^-1".format(
                  r["freq_mae_low_cm"], r["freq_mae_cm"], r["loss_cartesian"], r["noise_floor_cm"]))

        # must-pass: the base model against a Label that IS its own Hessian
        self_label = [a.copy() for a in ref]
        for a, src in zip(self_label, ref):
            a.calc = src.calc
            a.info = dict(src.info)
            a.info["hessian"] = np.asarray(mace_basin.info["hessian"])
        dd = Path(dataset_mod.datasets_dir(Path(tmp), "selfsmoke", "selfsmoke"))
        dd.mkdir(parents=True)
        dataset._write_split(dd / "test.{}.extxyz".format(LEVEL), [(a, "test") for a in self_label], reference=True)
        srows, _ = judge.frame_rows(dd, "selfsmoke", LEVEL, calc, splits=("test",), index=[])
        # not exactly zero: the Label is the engine's Hessian at the FIXTURE's positions,
        # while the engine recomputes at the extxyz's 8-decimal ones (5e-9 A) -- 3e-5 cm^-1
        check("A5 through the judge: H_r := H_base gives ~0 loss and ~0 frequency error",
              srows[0]["loss_cartesian"] < 1e-12 and srows[0]["freq_mae_cm"] < 1e-3, srows[0])

        ramp = dict(max_K=20.0, step_K=5.0, step_ps=0.05, seed=1)        # 4 stages x 50 steps: the machinery, not the physics
        closed = judge.run(Path(tmp), "smoke", "smoke", LEVEL, calc, name, base_calc=calc, base_engine=name,
                           run_name="base_closed", splits=("test",), write=True)
        check("the default judge run has the gate CLOSED (S0-C-60): VERDICT = REPORTED, GATE_OPEN false, every row still carries its result",
              closed["info"]["VERDICT"] == "REPORTED" and closed["info"]["GATE_OPEN"] is False
              and {l["LINE"]: l["RESULT"] for l in closed["verdict"]}["held_out_low_mode_mae_cm"] == "FAIL"
              and "CLOSED" in (closed["run_dir"] / "judge.out").read_text(encoding="utf-8"), closed["info"]["VERDICT"])
        out = judge.run(Path(tmp), "smoke", "smoke", LEVEL, calc, name, base_calc=calc, base_engine=name,
                        run_name="base", splits=("test",), write=True, ramp=ramp, gate=True)
        lines = {l["LINE"]: l for l in out["verdict"]}
        # The must-pass of the ticket is the NO-DEGRADATION line (base against base) and
        # the self-label zero above -- NOT the low-mode line. On this molecule the base
        # model FAILS the low-mode line at 12.7 cm^-1 against the 8.5 threshold, and that
        # is the judge working: 2-methyloxirane is an out-of-distribution ring, exactly
        # where MACE-OFF23's curvature error lives (S0-C-41: -57 cm^-1 on the lowest mode
        # of one of the three rings). A judge that passed the base model here would be
        # measuring nothing the fine-tune is for. Since ticket 22 that line is a REFERENCE
        # row: reported, never in the verdict.
        check("must-pass: judged against ITSELF the Hessian gate reads exactly 0 (ratio - 1), no gate row FAILs, VERDICT PASS",
              abs(lines["held_out_hessian_cartesian"]["VALUE"]) < 1e-12 and lines["held_out_hessian_cartesian"]["GATE"] == "yes"
              and all(l["RESULT"] != "FAIL" for l in out["verdict"] if l["GATE"] == "yes") and out["info"]["VERDICT"] == "PASS",
              [(l["LINE"], l["GATE"], l["RESULT"]) for l in out["verdict"]])
        check("the low-mode REFERENCE row is measured and the base model reads FAIL on it on this ring "
              "(12.7 cm^-1 against 8.5: the error the fine-tune exists to fix), without moving the verdict",
              lines["held_out_low_mode_mae_cm"]["GATE"] == "no" and lines["held_out_low_mode_mae_cm"]["RESULT"] == "FAIL"
              and lines["held_out_low_mode_mae_cm"]["VALUE"] > 8.5, lines["held_out_low_mode_mae_cm"])
        check("... and the line names the Label's own grid noise, so nobody reads a 12.7 "
              "against a 24 cm^-1 floor as settled",
              "grid noise" in lines["held_out_low_mode_mae_cm"]["NOTE"])
        check("... the Record is on disk (judge.out / .toml / .dat) with [[Displacement]] and [[Ramp]]",
              all((out["run_dir"] / ("judge" + e)).is_file() for e in (".out", ".toml", ".dat"))
              and len(prop.load(out["run_dir"] / "judge.toml")["Displacement"]) >= 2
              and len(prop.load(out["run_dir"] / "judge.toml")["Ramp"]) == 2)
        rec = prop.load(out["run_dir"] / "judge.toml")["Calculation_Info"]
        check("the judge Record carries the package identity: the distribution version and the package commit (40 hex or unknown)",
              isinstance(rec["HL_PACKAGE_VERSION"], str) and rec["HL_PACKAGE_VERSION"]
              and (rec["HL_PACKAGE_COMMIT"] == "unknown" or len(rec["HL_PACKAGE_COMMIT"]) == 40),
              (rec.get("HL_PACKAGE_VERSION"), rec.get("HL_PACKAGE_COMMIT")))
        disp_rows = {(x["DISTRIBUTION"], x["RMS_BIN"]): x for x in out["displacement"]}
        held = [x for k, x in disp_rows.items() if k[1] != "0"]
        check("the displacement bins: the basin bin (1 Hessian frame) and the held-out bins (4 E/F frames, no H sample), "
              "engine E/F errors equal the base's (it IS the base)",
              disp_rows[("out_of_molecule", "0")]["N_HESSIAN"] == 1 and sum(x["N_FRAMES"] for x in held) == 4
              and all(x["N_HESSIAN"] == 0 and x["HESSIAN_MAE"] is None for x in held)
              and all(abs(x["E_MAE_MEV_PER_ATOM"] - x["BASE_E_MAE_MEV_PER_ATOM"]) < 1e-9 for x in held)
              and all(x["F_RMSE_MEV_A"] > 0 for x in held), out["displacement"])
        print("    held-out bins on the base: " + "; ".join("{} {} frames E {:.2f} meV/atom F {:.1f} meV/A".format(
            x["RMS_BIN"], x["N_FRAMES"], x["E_MAE_MEV_PER_ATOM"], x["F_RMSE_MEV_A"]) for x in held))
        rr = {x["WHICH"]: x for x in out["ramp"]}
        check("the MD ramp (asked for explicitly) ran on the pinned molecule for engine and base (4 stages x 50 steps at 1 fs), "
              "both survive to 20 K, the reference line reads PASS (engine == base) and does not gate",
              set(rr) == {"engine", "base"} and all(x["SURVIVED"] and x["FAIL_T_K"] == 20.0 and x["N_STEPS"] == 200 for x in rr.values())
              and lines["md_ramp_K"]["GATE"] == "no" and lines["md_ramp_K"]["RESULT"] == "PASS"
              and out["info"]["RAMP_MAX_K"] == 20.0 and out["info"]["N_RAMP_MOLECULES"] == 1, (rr, lines.get("md_ramp_K")))
        print("    ramp: engine max ratio {:.3f} min ratio {:.3f} in {:.1f} s; base {:.3f} / {:.3f}".format(
            rr["engine"]["MAX_RATIO"], rr["engine"]["MIN_RATIO"], rr["engine"]["SECONDS"], rr["base"]["MAX_RATIO"], rr["base"]["MIN_RATIO"]))
        print("    verdict lines: " + ", ".join("{} [{}] {}".format(l["LINE"], l["GATE"], l["RESULT"]) for l in out["verdict"]))

        # must-fail: every frequency x0.9
        bad = judge.run(Path(tmp), "smoke", "smoke", LEVEL, judge.ScaledCalculator(calc, 0.9), name,
                        base_calc=calc, base_engine=name, run_name="base_x0.9", splits=("test",),
                        scale=0.9, write=True, ramp=None, gate=True)
        blines = {l["LINE"]: l for l in bad["verdict"]}
        check("must-fail: the 0.9x-scaled potential (Hessian x0.81) FAILs the Hessian GATE row and the VERDICT (S0-C-59); "
              "its low-mode reference row reads FAIL too",
              blines["held_out_hessian_cartesian"]["RESULT"] == "FAIL" and blines["held_out_hessian_cartesian"]["VALUE"] > 1.0
              and blines["held_out_low_mode_mae_cm"]["RESULT"] == "FAIL" and blines["held_out_low_mode_mae_cm"]["GATE"] == "no"
              and bad["info"]["VERDICT"] == "FAIL" and bad["info"]["VERDICT"] == judge.verdict_of(bad["verdict"]),
              [(l["LINE"], l["GATE"], l["RESULT"], l["VALUE"]) for l in bad["verdict"]])
        print("    scaled 0.9: low-mode MAE {:.2f} cm^-1 against the base model's {:.2f}".format(
            blines["held_out_low_mode_mae_cm"]["VALUE"], lines["held_out_low_mode_mae_cm"]["VALUE"]))
        check("... and its ||dH||^2/9N^2 is far worse than the base model's",
              bad["distributions"][0]["LOSS_CARTESIAN"] > 10 * out["distributions"][0]["LOSS_CARTESIAN"],
              (bad["distributions"][0]["LOSS_CARTESIAN"], out["distributions"][0]["LOSS_CARTESIAN"]))

        try:
            judge.run(Path(tmp), "nowhere", "nowhere", LEVEL, calc, name, write=False)
            check("a judge run with no labelled frame REFUSES (it must not answer PASS)", False)
        except ValueError as exc:
            check("a judge run with no labelled frame REFUSES (it must not answer PASS)",
                  "nothing to judge" in str(exc), str(exc))

    print("\n{} checks, {} failed".format(16, len(FAIL)))
    print("PASS" if not FAIL else "FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
