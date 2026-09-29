"""Ticket 13 of the Hessian-learning set: `openqha_hessian.run` fine-tunes the real
MACE-OFF23_medium through the mace fork, with the Hessian loss; tickets 18 and 21
(2026-09-21): the Replay that acts and is recorded, the Cartesian target, the
fixed-probe validation and the explicit control.

INTEGRATION (engine, CPU, minutes). Builds a two-molecule Dataset directory from the
2-methyloxirane frame fixture -- BASIN frames only in train and valid (S0-C-54), the
displaced frames of the fixture unused here -- then:

  * (21) a 4-epoch fine-tune with the default target (`cartesian`, `rademacher` k = 4):
    the run directory, the model file, the Record; the three validation curves exist
    (epoch -1 = the base model, then one per epoch); the epoch -1 Hessian value equals
    `smoke_fit.epoch_zero_balance`'s Cartesian L_H within 4 s.e. of the fixed-probe
    estimator (one frame, k = 4); Stage Two switches at epoch 3 and the log names the
    swa weights; the fine-tuned model loads in `MACECalculator`;
  * the validation estimator is the SAME on a second evaluate of the same model (the
    probes are fixed per frame) and `wants_hessian_at_eval` is False (no full matrix);
  * (18) a 3-epoch fine-tune with `--multiheads` and a two-frame Replay file with
    `config_weight = 3`: the Record's `PT_N_FRAMES` 2, `PT_CONFIG_WEIGHT` 3.0,
    `REPLAY_PER_HESSIAN_FRAME` 2, both heads' counts parsed from mace's log, the
    validation Hessian curve NOT constant (the term acts: commit C kept the loss) and
    `MACE_FORK_COMMIT` is not commit B's;
  * `--loss weighted` on the same files runs unchanged (the default path).

SKIPs without the model or without the fork.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tests/, where _testlib lives
from _testlib import openqha_src                                # noqa: E402

ROOT = openqha_src()
FIX = ROOT / "tests" / "data" / "methyloxirane_frames"
LEVEL = "wb97m-d3bj_def2-tzvppd"
#: the fork's commit B (the external-loss hook) in the rebuilt history; commit C follows it
FORK_COMMIT_B = "61582b0efb61cc251457c52367bb8239fe1cb2b5"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def build_dataset(tmp, name, level):
    """A Dataset directory with the merged MACE-form file: train = the basin frame,
    valid = the basin frame again (S0-C-54: basin frames only; the machinery, not
    generalisation). Two copies in each split so mace's batch of 2 is full.

    Every row carries its fixed probe set, as `dataset.build` writes it (S0-C-67): drawn
    from the frame's identity, stored in the file, read by the loss whenever it is in eval
    mode -- which mace also is on the TRAINING split, for its final error table.
    """
    from openqha.data import dataset, frames
    rows = []
    for i, split in enumerate(("train", "train", "valid", "valid")):
        basin = frames.read_frames(FIX / "basin.{}.extxyz".format(level))[0]   # re-read: copy() drops the calculator
        v = dataset.valid_probes(dataset.SEED, "fixture", ("basin", 0, i), len(basin))
        basin.info["valid_probes"] = v.reshape(-1)
        basin.info["has_valid_probes"] = True
        rows.append((basin, split))
    d = Path(tmp) / name
    d.mkdir(parents=True, exist_ok=True)
    dataset._write_split(dataset.merged_file(d, name, level), rows, reference=True)
    return d


def main():
    from openqha.potentials import engine
    from openqha_hessian import run as train_run
    try:
        import mace                                              # noqa: F401
        engine.model_path()
    except Exception as exc:                                     # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0
    fork = engine.mace_fork_info()
    if fork["mace_fork_commit"] == "unknown":
        print("SKIP: the installed mace is not the fork ({})".format(engine.MACE_FORK))
        return 0
    print("  mace {}  fork {}{}".format(__import__("mace").__version__, fork["mace_fork_commit"][:12],
                                        "  DIRTY" if fork["mace_fork_dirty"] else ""))
    import numpy as np
    import torch
    from ase.io import read, write as ase_write
    from mace.calculators import MACECalculator
    from openqha.store import property as prop
    from openqha_hessian import phl, phl_loss
    from openqha_hessian import smoke_fit

    with tempfile.TemporaryDirectory() as tmp:
        d = build_dataset(tmp, "smoke_fit", LEVEL)
        out = train_run.run_training(
            d, "test", "smoke_fit", LEVEL, "cart4", dry_run=True, strict_fork=False,
            hessian_weight=1e-3, max_epochs=4, batch_size=2)
        argv = out["argv"]
        check("dry run: the command names the external loss, the keys, float64, the cartesian target and the control",
              "--loss" in argv and argv[argv.index("--loss") + 1] == "external"
              and argv[argv.index("--loss_module") + 1] == train_run.LOSS_MODULE
              and argv[argv.index("--hessian_key") + 1] == "REF_hessian"
              and argv[argv.index("--default_dtype") + 1] == "float64"
              and argv[argv.index("--start_swa") + 1] == "3" and "--ema" in argv
              and argv[argv.index("--scheduler_patience") + 1] == "20", argv)
        check("dry run: the Record counts the frames and their Hessians (2 train / 2 valid, all labelled), no Replay",
              (out["info"]["N_TRAIN"], out["info"]["N_TRAIN_HESSIAN"]) == (2, 2)
              and (out["info"]["N_VALID"], out["info"]["N_VALID_HESSIAN"]) == (2, 2)
              and out["info"]["PT_N_FRAMES"] == 0 and out["info"]["REPLAY_PER_HESSIAN_FRAME"] == 0.0
              and out["info"]["VALID_PROBES"] == phl_loss.VALID_PROBES_LABEL, out["info"])
        check("dry run: the info the tests read carries the package identity too",
              "HL_PACKAGE_VERSION" in out["info"] and "HL_PACKAGE_COMMIT" in out["info"], sorted(out["info"]))
        check("dry run writes no model", not (out["run_dir"] / "cart4.model").is_file())

        # --- (21) the Cartesian target, four epochs, Stage Two at 3 ------------------------------
        out = train_run.run_training(
            d, "test", "smoke_fit", LEVEL, "cart4", strict_fork=False,
            hessian_weight=1e-3, max_epochs=4, batch_size=2, seed=7, device="cpu")
        info, run_dir = out["info"], out["run_dir"]
        check("the run produced a model file", Path(info["MODEL_FILE"]).is_file(), info["MODEL_FILE"])
        check("the Record is on disk (train.out / .toml / .dat)",
              all((run_dir / ("train" + e)).is_file() for e in (".out", ".toml", ".dat")))
        check("the epoch table has both splits", {r["split"] for r in out["epochs"]} >= {"train", "valid"},
              sorted({r["split"] for r in out["epochs"]}))
        check("every logged loss is finite", all(r["loss"] is None or r["loss"] == r["loss"] for r in out["epochs"]))
        curves = train_run.validation_curves(out["epochs"])
        check("the three validation curves exist: epoch -1 (the base) and epochs 0..3, every value finite",
              all([e for e, _v in curves[k]] == [-1, 0, 1, 2, 3] for k in ("valid_energy", "valid_forces", "valid_hessian"))
              and all(np.isfinite(v) for k in curves for _e, v in curves[k]), curves)
        check("the Hessian curve moved and the Record says so", info["HESSIAN_CURVE_MOVED"] is True
              and train_run.curve_moved(curves["valid_hessian"]), curves["valid_hessian"])
        log = "\n".join(p.read_text(errors="replace") for p in sorted((run_dir / "logs").glob("*.log")))
        check("Stage Two switched at epoch 3 and the log names the swa weights (hessian_weight 0.001 x 100 / 100)",
              info["STAGE_TWO_EPOCH"] == 3 and "Stage Two" in log and "hessian_weight=0.001" in log
              and abs(info["SWA_HESSIAN_WEIGHT"] - 1e-3) < 1e-15, (info["STAGE_TWO_EPOCH"], info["SWA_HESSIAN_WEIGHT"]))
        check("the Record carries the identities (foundation and model files, config SHA, fork commit) and the control",
              Path(info["FOUNDATION_FILE"]).is_file() and Path(info["MODEL_FILE"]).is_file()
              and info["FOUNDATION_FILE"] != info["MODEL_FILE"]
              and len(info["CONFIG_SHA256"]) == 64 and info["MACE_FORK_COMMIT"] != "unknown"
              and info["SECONDS_PER_EPOCH"] > 0 and info["LR"] == 0.01 and info["SWA_LR"] == 0.00025
              and info["EMA"] is True and info["SWA"] is True and info["START_SWA"] == 3, info)
        rec = prop.load(run_dir / "train.toml")["Calculation_Info"]
        check("the training Record on disk carries the package identity: the distribution version and the package "
              "commit (40 hex or unknown)",
              isinstance(rec["HL_PACKAGE_VERSION"], str) and rec["HL_PACKAGE_VERSION"]
              and (rec["HL_PACKAGE_COMMIT"] == "unknown" or len(rec["HL_PACKAGE_COMMIT"]) == 40)
              and rec["HL_PACKAGE_VERSION"] == info["HL_PACKAGE_VERSION"]
              and rec["HL_PACKAGE_COMMIT"] == info["HL_PACKAGE_COMMIT"],
              (rec.get("HL_PACKAGE_VERSION"), rec.get("HL_PACKAGE_COMMIT")))
        print("  {} epochs in {:.1f} s ({:.1f} s per epoch, {} train frames)".format(
            info["N_EPOCHS"], info["SECONDS"], info["SECONDS_PER_EPOCH"], info["N_TRAIN"]))
        for r in out["epochs"]:
            if r["split"] == "valid":
                print("    epoch {:3d} valid loss {}  E {}  F {}  H {}".format(
                    r["epoch"], "-" if r["loss"] is None else "{:.6f}".format(r["loss"]),
                    "-" if r["valid_energy"] is None else "{:.3e}".format(r["valid_energy"]),
                    "-" if r["valid_forces"] is None else "{:.3e}".format(r["valid_forces"]),
                    "-" if r["valid_hessian"] is None else "{:.3e}".format(r["valid_hessian"])))

        # the epoch -1 Hessian value is the base model's fixed-probe estimate of the Cartesian
        # target: within 4 s.e. of the exact epoch-zero balance (one frame, k = 4 Rademacher)
        base_calc = MACECalculator(model_paths=str(engine.model_path()), device="cpu", default_dtype="float64")
        bal = smoke_fit.epoch_zero_balance(base_calc, info["VALID_FILE"])
        atoms = read(str(info["VALID_FILE"]), index="0", format="extxyz")
        n3 = 3 * len(atoms)
        h_r = np.asarray(atoms.info["REF_hessian"], dtype=float).reshape(n3, n3)
        from openqha_hessian.judge import hessian_at
        h_e = hessian_at(base_calc, atoms)
        se = np.sqrt(phl.estimator_variance(h_e, h_r, k=4)["rademacher"])
        h_init = curves["valid_hessian"][0][1]
        check("epoch -1 validation Hessian term = epoch_zero_balance's Cartesian L_H within 4 s.e. ({:.3e} vs {:.3e}, s.e. {:.1e})".format(
              h_init, bal["L_H"], se), abs(h_init - bal["L_H"]) < 4 * se + 1e-12, (h_init, bal["L_H"], se))

        # the fine-tuned model loads and answers
        calc = MACECalculator(model_paths=info["MODEL_FILE"], device="cpu", default_dtype="float64")
        atoms.calc = calc
        e = float(atoms.get_potential_energy())
        h = calc.get_hessian(atoms)
        check("the fine-tuned model loads in MACECalculator and gives E and a (3N, N, 3) Hessian",
              e == e and h.shape == (3 * len(atoms), len(atoms), 3), (e, h.shape))

        # the validation term through mace's own evaluate: no full matrix, the force graph
        # kept, and the SAME value on a second pass (the probes are fixed per frame)
        from mace import data as mdata, tools as mtools
        from mace.tools import torch_geometric
        from mace.tools.train import evaluate
        torch.set_default_dtype(torch.float64)
        loss_fn = phl_loss.WeightedEnergyForcesHessianLoss(probe="rademacher", n_probes=4, seed=1)
        check("wants_hessian_at_eval False, wants_force_graph_at_eval True",
              loss_fn.wants_hessian_at_eval is False and loss_fn.wants_force_graph_at_eval is True)
        ks = mdata.KeySpecification.from_defaults()
        frames = read(str(info["VALID_FILE"]), index=":", format="extxyz")
        table = mtools.AtomicNumberTable(sorted({int(z) for a in frames for z in a.numbers}))
        ads = [mdata.AtomicData.from_config(mdata.config_from_atoms(a, key_specification=ks),
                                            z_table=table, cutoff=float(calc.r_max)) for a in frames]
        loader = torch_geometric.dataloader.DataLoader(dataset=ads, batch_size=2, shuffle=False)
        oa = {"energy": True, "forces": True, "virials": False, "stress": False,
              "hessian": loss_fn.wants_hessian_at_eval, "force_graph": loss_fn.wants_force_graph_at_eval}
        _l1, aux1 = evaluate(model=calc.models[0], loss_fn=loss_fn, data_loader=loader, output_args=oa, device=torch.device("cpu"))
        _l2, aux2 = evaluate(model=calc.models[0], loss_fn=loss_fn, data_loader=loader, output_args=oa, device=torch.device("cpu"))
        check("mace's evaluate: the estimator (no hessian_exact), the summary in aux, identical on two passes (1e-12)",
              loss_fn.last_terms["hessian_exact"] is None and aux1.get("valid_hessian_term") is not None
              and abs(aux1["valid_hessian_term"] - aux2["valid_hessian_term"]) < 1e-12
              and aux1["valid_hessian_n_labelled"] == 2, (aux1.get("valid_hessian_term"), aux2.get("valid_hessian_term")))
        print("    fixed-probe validation Hessian term on the fine-tuned model: {:.6e}".format(aux1["valid_hessian_term"]))

        entry = train_run.registry_entry(info)
        stamp = entry["name"].split("+", 1)[1]
        check("the ENGINES entry is a stamped fixed revision (mace_off23_<campaign>/<run>+<stamp>.model) with the "
              "index + config SHA as source and no fingerprint",
              entry["name"].startswith("test-cart4+") and entry["filename"] == "mace_off23_test/cart4+{}.model".format(stamp)
              and "config" in entry["source"] and "params_sha256" not in entry, entry)

        # --- (18) the Replay: two frames with config_weight 3, three epochs -------------------------
        pt = Path(tmp) / "pt.xyz"
        pt_valid = Path(tmp) / "pt.valid.xyz"
        pt_frames = read(str(ROOT / "tests" / "data" / "spice_tiny" / "train_large_neut_no_bad_clean.xyz"),
                         index=":", format="extxyz")
        for a in pt_frames:
            a.info["REF_energy"] = float(a.info.get("energy", -1.0))
            a.arrays["REF_forces"] = a.arrays.get("forces", a.get_forces())   # ASE may park them on the calculator
            a.info.pop("REF_hessian", None)
            a.info["config_weight"] = 3.0
        ase_write(str(pt), pt_frames[:2], format="extxyz")
        ase_write(str(pt_valid), pt_frames[2:3], format="extxyz")
        mh = train_run.run_training(
            d, "test", "smoke_fit", LEVEL, "mh1", strict_fork=False,
            hessian_weight="balance", max_epochs=3, batch_size=2, seed=7,
            multiheads=True, pt_train_file=str(pt), pt_valid_file=str(pt_valid))
        mi = mh["info"]
        check("w_H = balance (S0-C-60): the rule and the three base-model terms are in the Record and w_H = w_F L_F / L_H",
              mi["HESSIAN_WEIGHT_RULE"] == "balance" and mi["BALANCE_L_H"] > 0
              and abs(mi["HESSIAN_WEIGHT"] - 100.0 * mi["BALANCE_L_F"] / mi["BALANCE_L_H"]) < 1e-9
              and abs(mi["SWA_HESSIAN_WEIGHT"] - mi["HESSIAN_WEIGHT"]) < 1e-12, (mi["HESSIAN_WEIGHT"], mi["BALANCE_L_F"], mi["BALANCE_L_H"]))
        check("a multihead run completes and the Record says so: PT_N_FRAMES 2, PT_CONFIG_WEIGHT 3.0, 1 Replay frame per Hessian frame",
              mi["MULTIHEADS"] is True and mi["PT_TRAIN_FILE"] == str(pt) and mi["PT_VALID_FILE"] == str(pt_valid)
              and Path(mi["MODEL_FILE"]).is_file() and mi["PT_N_FRAMES"] == 2 and mi["PT_CONFIG_WEIGHT"] == "3.0"
              and mi["REPLAY_PER_HESSIAN_FRAME"] == 1.0 and mi["REAL_PT_DATA_RATIO_THRESHOLD"] == 0.0, mi)
        check("both heads' counts parsed from mace's log: pt 2 / 1, fine-tune 2 / 2",
              (mi["PT_HEAD_TRAIN"], mi["PT_HEAD_VALID"], mi["FT_HEAD_TRAIN"], mi["FT_HEAD_VALID"]) == (2, 1, 2, 2), mi)
        mcurves = train_run.validation_curves(mh["epochs"])
        check("with the Replay the validation Hessian curve is NOT constant: the term acts (commit C kept the loss)",
              mi["HESSIAN_CURVE_MOVED"] is True and len(mcurves["valid_hessian"]) >= 3, mcurves["valid_hessian"])
        mlog = "\n".join(p.read_text(errors="replace") for p in sorted((mh["run_dir"] / "logs").glob("*.log")))
        check("the log names the external loss in multihead mode and never the universal loss",
              "Multiheads finetuning with the external loss" in mlog and "WeightedEnergyForcesHessianLoss" in mlog
              and "UniversalLoss" not in mlog)
        check("MACE_FORK_COMMIT is not commit B's (commit C or later), 40 hex",
              mi["MACE_FORK_COMMIT"] != FORK_COMMIT_B and len(mi["MACE_FORK_COMMIT"]) == 40, mi["MACE_FORK_COMMIT"])
        check("... the replay frames carry no Hessian, so only the fine-tuning head's do",
              all("REF_hessian" not in a.info for a in read(str(pt), index=":", format="extxyz"))
              and mi["N_TRAIN_HESSIAN"] == 2)

        # --- ticket 34: the exact anchors (path A, S0-C-65) ---------------------------------------
        before, after = mi["VALID_HESSIAN_EXACT_BEFORE"], mi["VALID_HESSIAN_EXACT_AFTER"]
        exact_base = train_run.exact_valid_hessian("MACE-OFF23_medium", mi["VALID_FILE"])
        check("the Record carries both exact anchors, positive, and the BEFORE one is the base model's "
              "full-matrix Hessian term on the validation file (1e-10)",
              mi["EXACT_ANCHORS"] is True and before > 0 and after > 0
              and abs(before - exact_base) < 1e-10 * max(1.0, exact_base), (before, after, exact_base))
        probe_last = mi["VALID_HESSIAN_PROBE_LAST"]
        check("the last epoch's probe reading is recorded beside them and the relative distance is "
              "|probe - exact| / exact",
              probe_last > 0 and abs(mi["VALID_PROBE_OFFSET_RUN"] - abs(probe_last - after) / after) < 1e-12,
              (probe_last, after, mi["VALID_PROBE_OFFSET_RUN"]))
        check("the report states the pair and says the validation frames are not a generalisation reading",
              "EXACT ANCHORS" in (mh["run_dir"] / "train.out").read_text(encoding="utf-8")
              and "generalisation" in (mh["run_dir"] / "train.out").read_text(encoding="utf-8"))
        off = train_run.run_training(d, "test", "smoke_fit", LEVEL, "noanchor", strict_fork=False,
                                     hessian_weight=0.01, max_epochs=1, batch_size=2, seed=7,
                                     exact_anchors=False)
        check("--no-exact-anchors skips both readings (the keys stay at -1)",
              off["info"]["EXACT_ANCHORS"] is False and off["info"]["VALID_HESSIAN_EXACT_BEFORE"] == -1.0
              and off["info"]["VALID_HESSIAN_EXACT_AFTER"] == -1.0, off["info"]["VALID_HESSIAN_EXACT_BEFORE"])

        # the default path still works on the same files
        from mace.cli.run_train import run as mace_run
        from mace.tools import build_default_arg_parser
        plain = Path(tmp) / "plain"
        plain.mkdir()
        argv = train_run.mace_argv(info["TRAIN_FILE"], info["VALID_FILE"], "plain", plain,
                                   engine.model_path(), LEVEL, max_epochs=1, batch_size=2, seed=7, swa=False)
        i = argv.index("--loss")
        argv = argv[:i] + ["--loss", "weighted"] + argv[i + 4:]        # drop --loss external --loss_module
        args = build_default_arg_parser().parse_args(argv)
        import os
        cwd = os.getcwd()
        try:
            os.chdir(plain)
            mace_run(args)
        finally:
            os.chdir(cwd)
        check("the stock loss trains on the same files (the default path is untouched)",
              (plain / "plain.model").is_file() or list(plain.glob("*.model")))

    print("\n{} checks, {} failed".format(30, len(FAIL)))
    print("PASS" if not FAIL else "FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
