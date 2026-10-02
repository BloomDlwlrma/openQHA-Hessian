"""What `openqha_hessian.run` builds for mace,
what it refuses, and what its Record says -- no engine, no training.

Asserted: `mace_argv` names the external loss and the Dataset's keys, float64 and the
foundation's E0s, and carries the probe settings; `--multiheads` adds the replay flags
and nothing else does; `split_files` splits the merged Dataset file by its `split` key
and counts the Hessians; `check_fork` refuses a non-fork and a dirty checkout with a
message naming the fix, and diagnoses a `mace/` subdirectory of the working directory
as a shadow of the install; the Record's slim schema covers every key the file keeps
(the info dict may carry more, by design) and
`parse_results` / `parse_epochs` read mace's two output forms; `registry_entry` names
the index and the config file.

The Replay: the argv never contains `num_samples_pt`, contains
`--real_pt_data_ratio_threshold 0` with `--multiheads` and not without, passes
`--pt_valid_file`; `replay_file_summary` counts a two-frame file with `config_weight = 3`
and reads `3.0` (and `mixed`, and `1.0` without the key); `parse_head_counts` reads both
heads' lines; the schema has the four `PT_*` keys and no `NUM_SAMPLES_PT`; `run_training`
refuses `num_samples_pt`; `smoke_fit.replay_ratio` unchanged.

The control: `mace_argv` carries `--lr`, `--scheduler_patience 20`,
`--patience 50`, `--eval_interval 1`, `--ema`, `--swa --start_swa 3/4 --swa_lr lr/40`
and the Stage Two weights with `swa_hessian_weight = w_H x w_F^(2) / w_F` (the rule as
arithmetic); the target defaults to cartesian; `parse_results` / `parse_epochs` yield
the three validation curves and `parse_stage_two_epoch` the switch; `curve_moved` tells
a flat Hessian curve from a moving one.
"""
import os
import sys
import tempfile
from pathlib import Path

from openqha.store import property as prop
from openqha_hessian import run as train_run

FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def pairs(argv):
    """the settings as mace's parser sees them, bare flags included (`--save_cpu`)"""
    return {"--" + k: v for k, v in train_run.argv_pairs(argv).items()}


def main():
    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from openqha.data import dataset

    # --- the command line -----------------------------------------------------------------
    argv = train_run.mace_argv("tr.xyz", "va.xyz", "r1", "/tmp/run", "/w/base.model", "lvl",
                               hessian_weight=0.25, probe="gaussian", n_probes=7,
                               max_epochs=3, batch_size=2, seed=5, device="cuda")
    p = pairs(argv)
    check("the loss is ours, by name, through the fork's hook",
          p["--loss"] == "external" and p["--loss_module"] == "openqha_hessian.phl_loss:build"
          and p["--loss_module"].startswith("openqha_hessian"), p.get("--loss_module"))
    check("the keys are the Dataset's and the dtype is float64",
          (p["--energy_key"], p["--forces_key"], p["--hessian_key"]) == ("REF_energy", "REF_forces", "REF_hessian")
          and p["--default_dtype"] == "float64", p)
    check("the isolated-atom energies come from the foundation (the energy zero does not move)",
          p["--E0s"] == "foundation")
    check("the probe settings reach mace",
          (p["--hessian_weight"], p["--hessian_probe"], p["--n_hessian_probes"])
          == ("0.25", "gaussian", "7"), p)
    check("the loop settings reach mace",
          (p["--max_num_epochs"], p["--batch_size"], p["--seed"], p["--device"]) == ("3", "2", "5", "cuda"))
    check("without --multiheads the replay is off, no pt file is named and no duplication threshold is emitted",
          p["--multiheads_finetuning"] == "False" and "--pt_train_file" not in p
          and "--real_pt_data_ratio_threshold" not in p)
    check("num_samples_pt never appears in the argv", "--num_samples_pt" not in p)
    p2 = pairs(train_run.mace_argv("tr.xyz", "va.xyz", "r", "/tmp", "/w/b.model", "l",
                                   multiheads=True, pt_train_file="spice.xyz", pt_valid_file="spice.valid.xyz"))
    check("--multiheads adds the replay head, its file, its validation file and the threshold 0 -- and no sample count",
          p2["--multiheads_finetuning"] == "True" and p2["--pt_train_file"] == "spice.xyz"
          and p2["--pt_valid_file"] == "spice.valid.xyz" and p2["--real_pt_data_ratio_threshold"] == "0.0"
          and "--num_samples_pt" not in p2, p2)
    try:
        train_run.mace_argv("t", "v", "r", "/tmp", "/b", "l", num_samples_pt=5)
        check("mace_argv refuses num_samples_pt", False)
    except TypeError:
        check("mace_argv refuses num_samples_pt", True)
    extra = train_run.mace_argv("t", "v", "r", "/tmp", "/b", "l", extra=["--clip_grad", "1.0"])
    check("--mace-arg passes through as given", extra[-2:] == ["--clip_grad", "1.0"])
    extra_eq = train_run.mace_argv("t", "v", "r", "/tmp", "/b", "l", extra=["--clip_grad=1.0"])
    ep = train_run.argv_pairs(extra_eq)
    check("a single --key=value extra reads back as a pair (config.yaml truth)",
          extra_eq[-1] == "--clip_grad=1.0" and ep.get("clip_grad") == "1.0" and "clip_grad=1.0" not in ep, ep)
    check("a --key=value token splits at its first = only (= kept in the value, empty allowed)",
          train_run.argv_pairs(["--a=1", "--b=x=y", "--c", "--d="]) == {"a": "1", "b": "x=y", "c": True, "d": ""})
    # --- the control -------------------------------------------------------------------------------
    pc = pairs(train_run.mace_argv("t", "v", "r", "/tmp", "/b", "l", max_epochs=100, hessian_weight=0.02,
                                   forces_weight=100.0))
    check("the control defaults: lr 0.01, scheduler_patience 20, patience 50, eval_interval 1, ema, swa at 75, swa_lr 0.00025",
          (pc["--lr"], pc["--scheduler_patience"], pc["--patience"], pc["--eval_interval"]) == ("0.01", "20", "50", "1")
          and pc["--ema"] is True and pc["--swa"] is True and pc["--start_swa"] == "75" and pc["--swa_lr"] == "0.00025", pc)
    check("--ema_decay rides with --ema (mace's default 0.99)", pc["--ema_decay"] == "0.99", pc.get("--ema_decay"))
    check("the Stage Two weights: 1000 / 100 and w_H^(2) = 0.02 x 100 / 100 = 0.02",
          (pc["--swa_energy_weight"], pc["--swa_forces_weight"], pc["--swa_hessian_weight"]) == ("1000.0", "100.0", "0.02"), pc)
    check("no --hessian_mode_weighting is emitted: fork commit D took it out of the parser",
          "--hessian_mode_weighting" not in pc, sorted(pc))
    ctl = train_run.control_settings(8, lr=0.004, swa_forces_weight=10.0, hessian_weight=0.5, forces_weight=1000.0)
    check("control_settings as arithmetic: start_swa 6, swa_lr 1e-4, w_H^(2) = 0.5 x 10 / 1000 = 0.005",
          ctl["START_SWA"] == 6 and abs(ctl["SWA_LR"] - 1e-4) < 1e-15 and abs(ctl["SWA_HESSIAN_WEIGHT"] - 0.005) < 1e-15, ctl)
    check("start_swa is at least 1 (max_epochs 1); an explicit swa_hessian_weight wins",
          train_run.control_settings(1)["START_SWA"] == 1
          and train_run.stage_two_weights(0.5, 1000.0, 10.0, swa_hessian_weight=7.0) == 7.0)
    cm = train_run.control_settings(10, multiheads=True)
    check("the fork's multihead rule is mirrored: lr 0.0001, EMA on, decay 0.99999",
          (cm["LR"], cm["EMA"], cm["EMA_DECAY"]) == (0.0001, True, 0.99999), cm)
    check("... and swa_lr stays derived from the requested lr (the fork does not recompute it)",
          abs(cm["SWA_LR"] - 0.01 / 40) < 1e-15, cm)
    cf = train_run.control_settings(10, lr=0.002, ema=False, ema_decay=0.995, multiheads=True, force_mh_ft_lr=True)
    check("force_mh_ft_lr steps the mirror aside", (cf["LR"], cf["EMA"], cf["EMA_DECAY"]) == (0.002, False, 0.995), cf)
    changed = {}
    train_run.control_settings(10, lr=0.002, multiheads=True, warn=changed.update)
    check("the mirror reports what it replaced", changed.get("LR") == (0.002, 0.0001), changed)
    ovm = train_run.control_settings(10, multiheads=True, overrides={"LR": 0.002, "EMA_DECAY": 0.995})
    check("the extras fold sits under the fork rule: an overlaid lr is still forced in multihead",
          (ovm["LR"], ovm["EMA_DECAY"]) == (0.0001, 0.99999), ovm)
    ovs = train_run.control_settings(10, overrides={"LR": 0.002, "EMA_DECAY": 0.995})
    check("... while single-head keeps the overlaid values", (ovs["LR"], ovs["EMA_DECAY"]) == (0.002, 0.995), ovs)
    pmh = pairs(train_run.mace_argv("t", "v", "r", "/tmp", "/b", "l", multiheads=True))
    check("multihead emission carries the mirrored lr and decay, and no force flag",
          pmh["--lr"] == "0.0001" and pmh["--ema_decay"] == "0.99999" and "--force_mh_ft_lr" not in pmh, pmh)
    psh = pairs(train_run.mace_argv("t", "v", "r", "/tmp", "/b", "l", force_mh_ft_lr=True))
    check("a single-head run never carries the multihead force flag", "--force_mh_ft_lr" not in psh, psh)
    pf = pairs(train_run.mace_argv("t", "v", "r", "/tmp", "/b", "l", multiheads=True, force_mh_ft_lr=True))
    check("a forced run emits the native flag and keeps the requested lr",
          pf["--force_mh_ft_lr"] == "True" and pf["--lr"] == "0.01", pf)
    ov = train_run.control_overrides(["--clip_grad", "1.0", "--lr", "0.002", "--ema_decay=0.995", "--ema"])
    check("the extras fold takes the controlled keys only, in mace's spelling",
          ov == {"LR": 0.002, "EMA_DECAY": 0.995, "EMA": True}, ov)
    check("... and carries the force verdict",
          train_run.control_overrides(["--force_mh_ft_lr", "True"]) == {"FORCE_MH_FT_LR": True})
    try:
        train_run.run_training("t", "t", "t", "l", "r", multiheads=True)
        check("run_training refuses multiheads without a Replay file", False)
    except ValueError as exc:
        check("run_training refuses multiheads without a Replay file",
              "--pt-train-file" in str(exc) and "Replay" in str(exc), exc)
    epoch_schema = train_run.SCHEMA["Epoch"]
    check("the Epoch block the judge reads is intact: VALID_ENERGY / VALID_FORCES / VALID_HESSIAN typed",
          all(epoch_schema[k][0] == "Double" for k in ("VALID_ENERGY", "VALID_FORCES", "VALID_HESSIAN"))
          and epoch_schema["SPLIT"][0] == "String" and epoch_schema["EPOCH"][0] == "Integer")
    pn = pairs(train_run.mace_argv("t", "v", "r", "/tmp", "/b", "l", ema=False, swa=False))
    check("ema and swa can be switched off (no --ema, no --swa flags)", "--ema" not in pn and "--swa" not in pn and "--start_swa" not in pn)
    check("a bare flag maps to True, not to the next flag's name",
          train_run.argv_pairs(["--save_cpu", "--seed", "3"]) == {"save_cpu": True, "seed": "3"})

    # --- splitting the Dataset's merged file -------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.default_rng(0)
        rows = []
        for i, split in enumerate(["train", "train", "train", "valid", "test"]):
            at = Atoms("H2O", positions=rng.standard_normal((3, 3)))
            at.calc = SinglePointCalculator(at, energy=float(i), forces=np.zeros((3, 3)))
            at.info = dict(qm9_index="x", generator="basin", basin=0, k=i)
            if i % 2 == 0:                                    # frames 0, 2, 4 carry a Hessian
                h = np.eye(9)
                at.info["hessian"] = h
            rows.append((at, split))
        dataset._write_split(dataset.merged_file(td, "ds", "lvl"), rows, reference=True)
        files, counts = train_run.split_files(td, "ds", "lvl", td)
        check("split_files writes one file per split from the `split` key",
              files["train"].is_file() and files["valid"].is_file()
              and files["train"].name == "train.lvl.extxyz", files)
        check("... and counts the frames and the Hessians (train 3 / 2, valid 1 / 0)",
              counts == {"train": (3, 2), "valid": (1, 0)}, counts)
        try:
            train_run.split_files(td, "no_such", "lvl", td)
            check("a missing Dataset file is refused by name", False)
        except FileNotFoundError as exc:
            check("a missing Dataset file is refused by name", "04_dataset.py" in str(exc))

    # --- the fork guard -------------------------------------------------------------------------
    from openqha.potentials import engine
    real = engine.mace_fork_info
    try:
        engine.mace_fork_info = lambda: dict(mace_fork_commit="unknown", mace_fork_dirty=None, mace_fork_path=None)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            shadowed, plain = Path(td) / "shadowed", Path(td) / "plain"
            shadowed.mkdir()
            (shadowed / "mace").mkdir()          # a mace/ subdirectory shadows the editable install
            plain.mkdir()
            shadow_msg = plain_msg = "<no refusal>"
            try:
                os.chdir(str(shadowed))
                try:
                    train_run.check_fork(strict=True)
                except RuntimeError as exc:
                    shadow_msg = str(exc)
                os.chdir(str(plain))
                try:
                    train_run.check_fork(strict=True)
                except RuntimeError as exc:
                    plain_msg = str(exc)
            finally:
                os.chdir(cwd)
        check("a non-fork mace is refused, naming the install script and the fork identity",
              "pip install -e" in shadow_msg and "install.sh" in shadow_msg and engine.MACE_FORK in shadow_msg,
              shadow_msg)
        check("the shadow note appears with a mace/ subdirectory of the working directory, and not without",
              "cd into a repository root" in shadow_msg and "cd into a repository root" not in plain_msg,
              (shadow_msg[-140:], plain_msg[-140:]))
        check("... unless strict is off", train_run.check_fork(strict=False)["mace_fork_commit"] == "unknown")
        engine.mace_fork_info = lambda: dict(mace_fork_commit="a" * 40, mace_fork_dirty=True, mace_fork_path="/w/fork")
        try:
            train_run.check_fork(strict=True)
            check("a dirty checkout is refused", False)
        except RuntimeError as exc:
            check("a dirty checkout is refused", "uncommitted changes" in str(exc), str(exc))
    finally:
        engine.mace_fork_info = real

    # --- mace's two output forms ----------------------------------------------------------------
    results = ('{"mode": "eval", "epoch": null, "head": "pt_head", "loss": 9.0, "rmse_e_per_atom": 0.09, "rmse_f": 0.9}\n'
               '{"mode": "eval", "epoch": null, "head": "Default", "loss": 1.3, "rmse_e_per_atom": 0.03, "rmse_f": 0.6, '
               '"valid_energy_term": 2e-4, "valid_forces_term": 3e-3, "valid_hessian_term": 0.6}\n'
               '{"mode": "opt", "epoch": 0, "loss": 1.5, "rmse_e_per_atom": 0.02, "rmse_f": 0.5}\n'
               '{"mode": "eval", "epoch": 0, "head": "pt_head", "loss": 8.0, "rmse_e_per_atom": 0.08, "rmse_f": 0.8}\n'
               '{"mode": "eval", "epoch": 0, "loss": 1.2, "rmse_e_per_atom": 0.01, "rmse_f": 0.4, '
               '"valid_energy_term": 1e-4, "valid_forces_term": 2e-3, "valid_hessian_term": 0.5}\n'
               '{"mode": "eval", "epoch": 1, "loss": 1.1, "rmse_e_per_atom": 0.01, "rmse_f": 0.4, '
               '"valid_energy_term": 9e-5, "valid_forces_term": 1.9e-3, "valid_hessian_term": 0.4}\n'
               'not json\n')
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "run_train.txt").write_text(results)
        rows = train_run.parse_results(td)
    check("parse_results reads both splits, converts eV to meV, carries the three validation terms, keeps the "
          "initial evaluation as epoch -1 and drops the pretraining head's rows",
          [(r["epoch"], r["split"]) for r in rows] == [(-1, "valid"), (0, "train"), (0, "valid"), (1, "valid")]
          and rows[1]["rmse_f_meV_A"] == 500.0 and rows[2]["rmse_e_per_atom_meV"] == 10.0
          and (rows[2]["valid_energy"], rows[2]["valid_forces"], rows[2]["valid_hessian"]) == (1e-4, 2e-3, 0.5)
          and rows[0]["valid_hessian"] == 0.6 and rows[1]["valid_hessian"] is None, rows)
    curves = train_run.validation_curves(rows)
    check("validation_curves: one point per epoch per term (the initial one first); the Hessian curve moved",
          curves["valid_hessian"] == [(-1, 0.6), (0, 0.5), (1, 0.4)] and curves["valid_energy"] == [(-1, 2e-4), (0, 1e-4), (1, 9e-5)]
          and train_run.curve_moved(curves["valid_hessian"]), curves)
    check("curve_moved: a flat curve, a one-point curve and an empty one are False",
          not train_run.curve_moved([(0, 0.5), (1, 0.5), (2, 0.5 + 1e-9)]) and not train_run.curve_moved([(0, 0.5)])
          and not train_run.curve_moved([]))
    log = ("2026-09-20 21:09:00.000 INFO: openQHA loss (valid): energy=- forces=9.000000e-03 hessian=- n_labelled=0 "
           "probes=rademacher k=4 fixed target=cartesian\n"
           "2026-09-20 21:09:00.100 INFO: Initial: head: pt_head, loss=9.0, RMSE_E_per_atom=   90.00 meV, RMSE_F=   90.00 meV / A\n"
           "2026-09-20 21:09:01.000 INFO: openQHA loss (valid): energy=2.000000e-04 forces=3.000000e-03 hessian=6.000000e-01 "
           "n_labelled=3 probes=rademacher k=4 fixed target=cartesian\n"
           "2026-09-20 21:09:01.100 INFO: Initial: head: Default, loss=0.07, RMSE_E_per_atom=   30.00 meV, RMSE_F=   80.00 meV / A\n"
           "2026-09-20 21:09:03.100 INFO: openQHA loss (valid): energy=1.000000e-04 forces=2.000000e-03 "
           "hessian=5.000000e-01 n_labelled=3 probes=rademacher k=4 fixed target=cartesian\n"
           "2026-09-20 21:09:03.573 INFO: Epoch 1: head: Default, loss=0.05997006, "
           "RMSE_E_per_atom=   20.92 meV, RMSE_F=   77.16 meV / A\n"
           "2026-09-20 21:10:00.000 INFO: Changing loss based on Stage Two Weights\n"
           "2026-09-20 21:10:03.100 INFO: openQHA loss (valid): energy=9.000000e-05 forces=- hessian=4.000000e-01 "
           "n_labelled=3 probes=rademacher k=4 fixed target=cartesian\n"
           "2026-09-20 21:10:03.573 INFO: Epoch 2: head: Default, loss=0.04, "
           "RMSE_E_per_atom=   20.00 meV, RMSE_F=   70.00 meV / A\n")
    rows = train_run.parse_epochs(log)
    check("parse_epochs reads mace's log lines as a fallback (Initial = epoch -1, pt_head dropped), the loss's own line attached",
          len(rows) == 3 and [r["epoch"] for r in rows] == [-1, 1, 2] and abs(rows[1]["loss"] - 0.05997006) < 1e-12
          and rows[1]["rmse_f_meV_A"] == 77.16 and rows[1]["valid_hessian"] == 0.5 and rows[1]["valid_energy"] == 1e-4
          and rows[0]["valid_hessian"] == 0.6 and rows[0]["loss"] == 0.07
          and rows[2]["valid_hessian"] == 0.4 and rows[2]["valid_forces"] is None, rows)
    check("parse_stage_two_epoch: the first Epoch line after the switch (2); -1 without a switch",
          train_run.parse_stage_two_epoch(log) == 2 and train_run.parse_stage_two_epoch("Epoch 1: loss=1") == -1)
    heads_log = ("INFO: =============    Processing head Default     ===========\n"
                 "INFO: Total number of configurations: train=17132, valid=850, tests=[],\n"
                 "INFO: =============    Processing head pt_head     ===========\n"
                 "INFO: Total number of configurations: train=5000, valid=200, tests=[],\n"
                 "INFO: Total number of configurations in pretraining: train=5000, valid=200\n")
    heads = train_run.parse_head_counts(heads_log)
    check("parse_head_counts: both heads' train / valid and the pretraining summary line",
          heads == {"Default": (17132, 850), "pt_head": (5000, 200), "pt": (5000, 200)}, heads)

    # --- the Replay file ------------------------------------------------------------------------
    from ase.io import write as ase_write
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fr = []
        for i in range(2):
            at = Atoms("H2O", positions=np.random.default_rng(i).standard_normal((3, 3)))
            at.info["REF_energy"] = -1.0 - i
            at.arrays["REF_forces"] = np.zeros((3, 3))
            at.info["config_weight"] = 3.0
            fr.append(at)
        ase_write(str(td / "pt.xyz"), fr, format="extxyz")
        rs = train_run.replay_file_summary(td / "pt.xyz")
        check("replay_file_summary: two frames, config_weight 3.0 read from the headers",
              rs["n_frames"] == 2 and rs["config_weight"] == "3.0" and rs["weights"] == {3.0}, rs)
        fr[1].info["config_weight"] = 10.0
        ase_write(str(td / "mixed.xyz"), fr, format="extxyz")
        for a in fr:
            a.info.pop("config_weight")
        ase_write(str(td / "plain.xyz"), fr, format="extxyz")
        check("... `mixed` for two values, `1.0` (mace's default) when no frame carries the key",
              train_run.replay_file_summary(td / "mixed.xyz")["config_weight"] == "mixed"
              and train_run.replay_file_summary(td / "plain.xyz")["config_weight"] == "1.0")
    from openqha_hessian import smoke_fit
    rr = smoke_fit.replay_ratio(300, 100, 5000)
    check("smoke_fit.replay_ratio unchanged: 5000 frames over 100 Hessian frames = 50 per Hessian frame",
          rr["REPLAY_PER_HESSIAN_FRAME"] == 50.0 and rr["PFT_REFERENCE"] == 4.0)
    try:
        train_run.run_training("/no/such", "t", "n", "l", "r", num_samples_pt=5)
        check("run_training refuses num_samples_pt with the draw tool named", False)
    except TypeError as exc:
        check("run_training refuses num_samples_pt with the draw tool named", "s0_spice_pt_draw" in str(exc))

    # --- the balance helper's pass-through (no engine: the calculator and the balance are patched) --
    from openqha_hessian import smoke_fit as sf
    calls = []
    real_calc, real_balance = engine.calculator, sf.epoch_zero_balance

    def fake_balance(calc, train_file, **kw):
        calls.append(dict(kw))
        return dict(PROBE=kw.get("probe"), N_PROBES=kw.get("n_probes"), L_E=1.0, L_F=2.0, L_H=0.5,
                    HESSIAN_WEIGHT_BALANCED=400.0)

    engine.calculator = lambda device="cpu", name=None: (object(), name, {})
    sf.epoch_zero_balance = fake_balance
    try:
        b = train_run.hessian_weight_balance("MACE-OFF23_medium", "tr.xyz", probe="rademacher", n_probes=7, seed=42)
        b_def = train_run.hessian_weight_balance("MACE-OFF23_medium", "tr.xyz")
    finally:
        engine.calculator, sf.epoch_zero_balance = real_calc, real_balance
    check("hessian_weight_balance passes the run's probe kind, k and seed through to epoch_zero_balance "
          "(explicit rademacher k=7 seed=42; defaults gaussian k=4 seed=123)",
          [(c.get("probe"), c.get("n_probes"), c.get("seed")) for c in calls] == [("rademacher", 7, 42), ("gaussian", 4, 123)]
          and (b["PROBE"], b["N_PROBES"]) == ("rademacher", 7) and b_def["N_PROBES"] == 4,
          (calls, (b["PROBE"], b["N_PROBES"])))

    # --- the Record's slim schema covers what the file keeps (user ruling 2026-10-02) --------------
    written = {"RUN", "TAG", "NAME", "LEVEL", "DATASET_DIR", "INDEX_FILE", "TRAIN_FILE", "VALID_FILE",
               "N_TRAIN", "N_TRAIN_HESSIAN", "N_VALID", "N_VALID_HESSIAN", "FOUNDATION_MODEL",
               "FOUNDATION_FILE", "MODEL_FILE", "CONFIG_FILE", "LOSS",
               "ENERGY_WEIGHT", "FORCES_WEIGHT", "HESSIAN_WEIGHT", "HESSIAN_WEIGHT_RULE",
               "BALANCE_L_E", "BALANCE_L_F", "BALANCE_L_H", "BALANCE_PROBE", "BALANCE_N_PROBES",
               "PROBE", "N_PROBES", "MAX_NUM_EPOCHS", "BATCH_SIZE", "SEED", "DEVICE", "DTYPE",
               "MULTIHEADS", "PT_TRAIN_FILE", "PT_VALID_FILE", "PT_N_FRAMES", "PT_CONFIG_WEIGHT",
               "STAGE_TWO_EPOCH", "HESSIAN_CURVE_MOVED", "N_EPOCHS", "SECONDS", "SECONDS_PER_EPOCH",
               "EXACT_ANCHORS", "VALID_HESSIAN_EXACT_BEFORE", "VALID_HESSIAN_EXACT_AFTER",
               "VALID_HESSIAN_PROBE_LAST", "VALID_PROBE_OFFSET_RUN",
               "MACE_VERSION", "MACE_FORK", "MACE_FORK_COMMIT",
               "HL_PACKAGE_VERSION", "HL_PACKAGE_COMMIT"}
    schema = set(train_run.SCHEMA["Calculation_Info"])
    check("every kept key is in the schema", written <= schema, sorted(written - schema))
    check("the stale NUM_SAMPLES_PT key is gone from the schema; the PT_* keys are there",
          "NUM_SAMPLES_PT" not in schema and {"PT_N_FRAMES", "PT_CONFIG_WEIGHT", "PT_TRAIN_FILE", "PT_VALID_FILE"} <= schema)
    sc = train_run.SCHEMA["Calculation_Info"]
    check("the balance provenance fields are typed and the amended descriptions carry the estimator reading "
          "(measured under the run's probe setting; pre-amendment Records hold the exact full-matrix value)",
          sc["BALANCE_PROBE"][0] == "String" and sc["BALANCE_N_PROBES"][0] == "Integer"
          and "probe setting" in sc["HESSIAN_WEIGHT_RULE"][2] and "BALANCE_PROBE" in sc["HESSIAN_WEIGHT_RULE"][2]
          and "probe setting" in sc["BALANCE_L_H"][2] and "before the estimator amendment" in sc["BALANCE_L_H"][2],
          (sc.get("BALANCE_PROBE"), sc.get("BALANCE_N_PROBES")))
    check("the retired keys are gone from the schema (the weight identities, the config SHA, mace's head counts, "
          "the control block, the validation-probe label)",
          not {"FOUNDATION_PARAMS_SHA256", "MODEL_PARAMS_SHA256", "MODEL_N_TENSORS", "ENGINE_PARAMS_SHA256",
               "CONFIG_SHA256",
               "VALID_PROBES", "PT_HEAD_TRAIN", "PT_HEAD_VALID", "FT_HEAD_TRAIN", "FT_HEAD_VALID",
               "REPLAY_PER_HESSIAN_FRAME", "REAL_PT_DATA_RATIO_THRESHOLD",
               "LR", "SCHEDULER_PATIENCE", "PATIENCE", "EVAL_INTERVAL", "EMA", "EMA_DECAY",
               "SWA", "START_SWA", "SWA_LR", "SWA_ENERGY_WEIGHT", "SWA_FORCES_WEIGHT",
               "SWA_HESSIAN_WEIGHT"} & schema)
    check("the module is hash-free: no hashlib import, no digest helper (the checksum machinery is out)",
          "hashlib" not in Path(train_run.__file__).read_text(encoding="utf-8")
          and not hasattr(train_run, "sha256_file"))

    info = dict(FOUNDATION_MODEL="MACE-OFF23_medium", RUN="w1", INDEX_FILE="/r/index.dat",
                CONFIG_FILE="/r/train/w1/config.yaml", TAG="draw300", NAME="draw300",
                N_TRAIN=90, N_TRAIN_HESSIAN=30,
                HESSIAN_WEIGHT=0.01, PROBE="rademacher", N_PROBES=4,
                MACE_FORK_COMMIT="b" * 40, MULTIHEADS=True, PT_N_FRAMES=5000,
                REPLAY_PER_HESSIAN_FRAME=5000 / 30, PT_CONFIG_WEIGHT="1.0")
    e = train_run.registry_entry(info, stamp="20260927-101530")
    check("registry_entry: a stamped fixed revision (mace_off23_<campaign>/<run>+<stamp>.model), the index + "
          "the config file the Record names as source, the Replay named, and no fingerprint",
          e["name"] == "draw300-w1+20260927-101530"
          and e["filename"] == "mace_off23_draw300/w1+20260927-101530.model"
          and e["source"] == "/r/index.dat + config /r/train/w1/config.yaml"
          and "params_sha256" not in e and "w_H 0.01" in e["note"] and "Replay 5000 frames" in e["note"]
          and "the full Cartesian matrix" in e["note"], e)

    # --- the Record writes and reads back ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        full = dict(info)
        for k in schema - set(full) - {"PROGNAME", "VERSION", "STATUS"}:
            full[k] = 0 if "N_" in k or "SECONDS" in k or "WEIGHT" in k else "x"
        for k in ("MULTIHEADS", "EMA", "SWA", "HESSIAN_CURVE_MOVED", "EXACT_ANCHORS"):
            full[k] = False
        for k in ("VALID_HESSIAN_EXACT_BEFORE", "VALID_HESSIAN_EXACT_AFTER", "VALID_HESSIAN_PROBE_LAST",
                  "VALID_PROBE_OFFSET_RUN"):
            full[k] = -1.0
        for k in ("PT_CONFIG_WEIGHT", "VALID_PROBES"):
            full[k] = "x"
        train_run.write_record(td, full, [dict(epoch=0, split="train", loss=1.0,
                                               rmse_e_per_atom_meV=2.0, rmse_f_meV_A=3.0),
                                          dict(epoch=0, split="valid", loss=1.0, rmse_e_per_atom_meV=2.0,
                                               rmse_f_meV_A=3.0, valid_energy=1e-4, valid_forces=2e-3,
                                               valid_hessian=0.5)])
        rec = prop.load(Path(td) / "train.toml")
        check("train.toml round-trips with NORMAL TERMINATION; the Epoch block keeps only the valid rows "
              "and an info key outside the schema (VALID_PROBES) is dropped from the file",
              rec["Calculation_Status"]["STATUS"] == prop.NORMAL_TERMINATION
              and rec["Calculation_Info"]["RUN"] == "w1" and len(rec["Epoch"]) == 1
              and rec["Epoch"][0]["SPLIT"] == "valid" and rec["Epoch"][0]["VALID_HESSIAN"] == 0.5
              and "VALID_PROBES" not in rec["Calculation_Info"], rec.get("Calculation_Status"))
        check("the Record is train.toml only (no train.out report, no train.dat table)",
              not (Path(td) / "train.out").exists() and not (Path(td) / "train.dat").exists())
        old = (Path(td) / "train.toml").read_text(encoding="utf-8").replace(
            "[Calculation_Info]", '[Calculation_Info]\nCONFIG_SHA256 = "{}"'.format("0" * 64), 1)
        (Path(td) / "train_old.toml").write_text(old, encoding="utf-8")
        rec_old = prop.load(Path(td) / "train_old.toml")["Calculation_Info"]
        check("a Record written before the checksum retirement still reads: its CONFIG_SHA256 is carried, ignored "
              "by the reader, and the field is not in the new schema",
              rec_old["CONFIG_SHA256"] == "0" * 64 and rec_old["RUN"] == "w1" and "CONFIG_SHA256" not in schema)

    print("\n{} checks, {} failed".format(45, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
