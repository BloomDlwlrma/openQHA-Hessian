"""The fine-tune: openQHA's own entry point into the mace fork's training loop.

PRODUCTION.

The shape is mace-md's: an entry point of OUR OWN builds mace's argument namespace and
calls `mace.cli.run_train.run(args)`. Nothing here reimplements a training loop; what
mace could not do -- read a Hessian label (commit A), take a loss from outside (commit
B), keep that loss in multihead mode and give it the force graph at evaluation (commit
C) -- lives in the fork, and the loss itself is `openqha_hessian.phl_loss`, reached by
name:

    --loss external --loss_module openqha_hessian.phl_loss:build

One run is a directory of the Dataset:

    <root>/<tag>/_datasets/<name>/train/<run>/
        config.yaml        every mace argument, as given (the file mace itself can re-run)
        <run>.model        the fine-tuned potential (and <run>_stagetwo.model with --swa)
        checkpoints/, logs/, results/    mace's own
        train.{out,toml}   the Record: the settings, the epoch table, the identities

THE REPLAY. `--multiheads` concatenates a file of the
base model's own training frames (`s0_spice_pt_draw.py` writes it) into the training set
as mace's `pt_head`. Its size is the FILE's: mace's `--num_samples_pt` is read only on
the Materials-Project download path and is never emitted here; `--real_pt_data_ratio_
threshold` is emitted as 0, because its default 0.1 silently duplicates the fine-tune
frames whenever they are fewer than a tenth of the replay. The Record counts the file's
frames (`PT_N_FRAMES`), reads their `config_weight` (`PT_CONFIG_WEIGHT`), parses both
heads' counts from mace's log and prints the ratio as replay frames per Hessian frame
(`REPLAY_PER_HESSIAN_FRAME`), the number the S0 scan rows R0-R4 were defined by. mace takes
`--valid_fraction` of the replay file for the pretraining head's OWN validation unless
`--pt_valid_file` names one; the draw tool writes that companion file and it is passed.

THE WEIGHT. `hessian_weight="balance"` makes the driver measure, on the BASE
model and the run's own train file before the first step, the epoch-0 balance
`w_H = w_F L_F / L_H` with `L_H` measured under the run's probe setting
(`smoke_fit.epoch_zero_balance`; `BALANCE_PROBE` / `BALANCE_N_PROBES` in the Record name
the estimator) and train with that value; the Record keeps the rule, the
three terms and the resolved `HESSIAN_WEIGHT`. A number is used as given.

THE CONTROL (the base's recipe, MACE-OFF23): every flag that steers the loop
is emitted explicitly and recorded -- `--lr`, `--scheduler_patience`, `--patience`,
`--eval_interval`, `--ema`, `--swa --start_swa --swa_lr` with the Stage Two weights
(`--swa_energy_weight`, `--swa_forces_weight`, `--swa_hessian_weight` by the rule
w_H x swa_forces_weight / forces_weight). The three validation curves (E, F, Hessian
terms per epoch, the loss's `eval_summary` through commit C) are parsed from mace's
`results/*.txt` (or its log) into the Record; a flat Hessian curve is a warning line.

The Record is what makes a fine-tuned potential traceable: the Dataset and its index, the
loss settings, the mace fork's commit, the base potential's resolved file and the
fine-tuned one's, the SHA-256 of the config file, and this package's own version and
commit. `05_train.py --register` turns those into an `ENGINES` entry.

Refused, not warned: training against a mace that is not the fork (`mace_fork_commit`
"unknown"), or against a dirty checkout. A model whose loss cannot be reproduced from a
commit is not a product.
"""
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openqha.potentials import engine
from openqha.store import dat, property as prop, report
from openqha_hessian import package_identity
from openqha_hessian import phl_loss

PROGNAME = "openQHA hl_train"
STEP = "train"

#: the loss module the fork's `--loss external` imports
LOSS_MODULE = "openqha_hessian.phl_loss:build"

#: MACE-torch's keys in the Dataset's merged file (`dataset.REF_*_KEY`)
ENERGY_KEY, FORCES_KEY, HESSIAN_KEY = "REF_energy", "REF_forces", "REF_hessian"

#: the control defaults (the base's recipe: scheduler_patience 20, patience 50,
#: swa_lr = lr / 40, Stage Two at 3/4 of the epochs; mace's Stage Two weights 1000 / 100)
DEFAULT_LR = 0.01
DEFAULT_SCHEDULER_PATIENCE = 20
DEFAULT_PATIENCE = 50
DEFAULT_EVAL_INTERVAL = 1
SWA_LR_RATIO = 40.0
DEFAULT_SWA_ENERGY_WEIGHT = 1000.0
DEFAULT_SWA_FORCES_WEIGHT = 100.0
#: mace's `--real_pt_data_ratio_threshold`: 0 = never duplicate the fine-tune frames
REAL_PT_DATA_RATIO_THRESHOLD = 0.0
#: a validation Hessian curve whose relative range is below this "did not move"
FLAT_CURVE_TOL = 1e-6

SCHEMA = {
    "Calculation_Info": {
        "PROGNAME": ("String", None, "the step that wrote this file"),
        "VERSION": ("String", None, "openQHA version"),
        "STATUS": ("String", None, "the completion marker"),
        "RUN": ("String", None, "the run name; the directory is <dataset>/train/<run>/"),
        "TAG": ("String", None, "the campaign tag"),
        "NAME": ("String", None, "the Dataset name"),
        "LEVEL": ("String", None, "the reference level of the labels"),
        "DATASET_DIR": ("String", None, "the Dataset directory"),
        "INDEX_FILE": ("String", None, "index.dat of the Dataset: which frame is in which split"),
        "TRAIN_FILE": ("String", None, "the extxyz mace trained on"),
        "VALID_FILE": ("String", None, "the extxyz mace validated on"),
        "N_TRAIN": ("Integer", None, "frames in the training file"),
        "N_VALID": ("Integer", None, "frames in the validation file"),
        "N_TRAIN_HESSIAN": ("Integer", None, "of those, frames carrying a reference Hessian"),
        "N_VALID_HESSIAN": ("Integer", None, "of those, frames carrying a reference Hessian"),
        "FOUNDATION_MODEL": ("String", None, "the base potential fine-tuned"),
        "FOUNDATION_FILE": ("String", None, "the resolved weight file of the base potential"),
        "MODEL_FILE": ("String", None, "the fine-tuned potential"),
        "CONFIG_FILE": ("String", None, "the mace argument file"),
        "CONFIG_SHA256": ("String", None, "SHA-256 of the config file: the training recipe's identity"),
        "LOSS": ("String", None, "the loss module and factory (mace's --loss external)"),
        "ENERGY_WEIGHT": ("Double", None, "w_E of eq. 11 (Stage One)"),
        "FORCES_WEIGHT": ("Double", None, "w_F of eq. 11 (Stage One)"),
        "HESSIAN_WEIGHT": ("Double", None, "w_H of eq. 11 (Stage One); the resolved value when the rule is balance"),
        "HESSIAN_WEIGHT_RULE": ("String", None, "given (a number on the command line) or balance (w_F L_F / L_H on the base model over the train file, L_H measured with the run's probe setting -- BALANCE_PROBE / BALANCE_N_PROBES name the estimator; the exact full-matrix reading stays the job of the anchors and the judge)"),
        "BALANCE_L_E": ("Double", None, "the base model's per-atom energy MSE on the train file (balance rule only, else 0)"),
        "BALANCE_L_F": ("Double", None, "the base model's force MSE on the train file (balance rule only, else 0)"),
        "BALANCE_L_H": ("Double", None, "the base model's Hessian loss ||dH||^2/(9N^2) on the train file's Hessian frames, measured with the run's probe setting (BALANCE_PROBE / BALANCE_N_PROBES; balance rule only, else 0); Records written before the estimator amendment hold the exact full-matrix reading"),
        "BALANCE_PROBE": ("String", None, "the estimator behind BALANCE_L_H: the run's probe setting (gaussian / rademacher / cartesian); - when the rule is given"),
        "BALANCE_N_PROBES": ("Integer", None, "probes per frame behind BALANCE_L_H (k of the estimator; 0 for the exact cartesian path or when the rule is given)"),
        "PROBE": ("String", None, "gaussian (PHL's Algorithm 1, the default) / rademacher / cartesian (Algorithm 2): the training probes"),
        "N_PROBES": ("Integer", None, "probes per structure per step (k of eq. 6)"),
        "VALID_PROBES": ("String", None, "the validation estimator: k fixed probes per frame, stored by the Dataset"),
        "MAX_NUM_EPOCHS": ("Integer", None, "epochs asked for"),
        "N_EPOCHS": ("Integer", None, "epochs the log holds"),
        "BATCH_SIZE": ("Integer", None, "structures per step"),
        "SEED": ("Integer", None, "mace's seed; also the training probe generator's"),
        "DEVICE": ("String", None, "cpu / cuda"),
        "DTYPE": ("String", None, "float64 throughout, as the Labels are"),
        "LR": ("Double", None, "the learning rate (mace's --lr)"),
        "SCHEDULER_PATIENCE": ("Integer", None, "ReduceLROnPlateau patience on the total validation loss, epochs"),
        "PATIENCE": ("Integer", None, "early-stopping patience on the total validation loss, epochs"),
        "EVAL_INTERVAL": ("Integer", None, "validate every this many epochs"),
        "EMA": ("Boolean", None, "exponential moving average of the parameters (mace's --ema)"),
        "SWA": ("Boolean", None, "Stage Two on (mace's --swa)"),
        "START_SWA": ("Integer", None, "the epoch Stage Two starts at (3/4 of MAX_NUM_EPOCHS by default)"),
        "SWA_LR": ("Double", None, "the Stage Two learning rate (LR / 40 by default, the base's ratio)"),
        "SWA_ENERGY_WEIGHT": ("Double", None, "w_E in Stage Two"),
        "SWA_FORCES_WEIGHT": ("Double", None, "w_F in Stage Two"),
        "SWA_HESSIAN_WEIGHT": ("Double", None, "w_H in Stage Two = HESSIAN_WEIGHT x SWA_FORCES_WEIGHT / FORCES_WEIGHT"),
        "STAGE_TWO_EPOCH": ("Integer", None, "the epoch the log switched to Stage Two at, or -1"),
        "MULTIHEADS": ("Boolean", None, "a Replay concatenated as mace's pretraining head"),
        "PT_TRAIN_FILE": ("String", None, "the Replay file, or -"),
        "PT_VALID_FILE": ("String", None, "the Replay's own validation file (mace takes --valid_fraction of the Replay without one), or -"),
        "PT_N_FRAMES": ("Integer", None, "frames counted in the Replay file (0 without one)"),
        "PT_CONFIG_WEIGHT": ("String", None, "the Replay frames' config_weight: one value, `mixed`, or - "),
        "PT_HEAD_TRAIN": ("Integer", None, "mace's count of pretraining-head training frames (parsed from its log; -1 when not logged)"),
        "PT_HEAD_VALID": ("Integer", None, "mace's count of pretraining-head validation frames (-1 when not logged)"),
        "FT_HEAD_TRAIN": ("Integer", None, "mace's count of fine-tuning-head training frames (-1 when not logged)"),
        "FT_HEAD_VALID": ("Integer", None, "mace's count of fine-tuning-head validation frames (-1 when not logged)"),
        "REPLAY_PER_HESSIAN_FRAME": ("Double", None, "PT_N_FRAMES / N_TRAIN_HESSIAN: the Replay's size as the S0 scan rows define it (0 without a Replay)"),
        "REAL_PT_DATA_RATIO_THRESHOLD": ("Double", None, "mace's duplication threshold, always 0 here (never duplicate the fine-tune frames)"),
        "VALID_HESSIAN_EXACT_BEFORE": ("Double", "eV^2/A^4", "the EXACT Hessian term on the validation file for the BASE model, from the full matrix; -1 when not measured"),
        "VALID_HESSIAN_EXACT_AFTER": ("Double", "eV^2/A^4", "the same quantity for the fine-tuned model: the pair says what the run moved on the target, with no estimator noise; -1 when not measured"),
        "VALID_HESSIAN_PROBE_LAST": ("Double", "eV^2/A^4", "the last epoch's in-loop reading of the same quantity (4 fixed probes per frame); -1 when absent"),
        "VALID_PROBE_OFFSET_RUN": ("Double", None, "|probe - exact| / exact at the end of the run: what the fixed-probe estimator cost on this validation set; -1 when either is missing"),
        "EXACT_ANCHORS": ("Boolean", None, "the two exact readings were taken (--no-exact-anchors turns them off)"),
        "HESSIAN_CURVE_MOVED": ("Boolean", None, "the validation Hessian term changed across the epochs (a flat curve means the term did not act)"),
        "MACE_VERSION": ("String", None, "mace.__version__ (the fork: 0.3.16+openqha)"),
        "MACE_FORK": ("String", None, "the fork the training code came from"),
        "MACE_FORK_COMMIT": ("String", None, "commit of the mace checkout (refused when unknown or dirty)"),
        "HL_PACKAGE_VERSION": ("String", None, "the openqha-hessian distribution's version (unknown when unreadable)"),
        "HL_PACKAGE_COMMIT": ("String", None, "commit of the package checkout, best-effort (unknown when unreadable)"),
        "SECONDS": ("Double", "s", "wall time of the whole run"),
        "SECONDS_PER_EPOCH": ("Double", "s", "wall time / epochs"),
    },
    "Epoch": {
        "EPOCH": ("Integer", None, "epoch index as mace logged it; -1 = the evaluation before any training (the base model)"),
        "SPLIT": ("String", None, "train (the step's loss) or valid (mace's evaluation)"),
        "LOSS": ("Double", None, "the total loss of eq. 11"),
        "RMSE_E_PER_ATOM_MEV": ("Double", "meV", "energy RMSE per atom, when the log gives it"),
        "RMSE_F_MEV_A": ("Double", "meV/A", "force RMSE, when the log gives it"),
        "VALID_ENERGY": ("Double", None, "the energy term of eq. 11 over the validation pass (valid rows)"),
        "VALID_FORCES": ("Double", None, "the forces term over the validation pass (valid rows)"),
        "VALID_HESSIAN": ("Double", None, "the Hessian term over the validation pass, the fixed-probe estimator (valid rows)"),
    },
}

EPOCH_ROW = {
    "epoch": ("Integer", None, "epoch index (-1 = before any training)"),
    "split": ("String", None, "train / valid"),
    "loss": ("Double", None, "total loss"),
    "rmse_e_per_atom_meV": ("Double", "meV", "energy RMSE per atom"),
    "rmse_f_meV_A": ("Double", "meV/A", "force RMSE"),
    "valid_energy": ("Double", None, "energy term over the validation pass"),
    "valid_forces": ("Double", None, "forces term over the validation pass"),
    "valid_hessian": ("Double", None, "Hessian term over the validation pass (fixed probes)"),
}

EPOCH_KEYS = ("epoch", "split", "loss", "rmse_e_per_atom_meV", "rmse_f_meV_A",
              "valid_energy", "valid_forces", "valid_hessian")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_fork(strict=True):
    """The mace that will train, as an identity. Refuses a non-fork or a dirty checkout
    (`strict`), because a loss that cannot be reproduced from a commit is not a product."""
    import mace
    info = engine.mace_fork_info()
    info["mace_version"] = mace.__version__
    if strict:
        if info["mace_fork_commit"] == "unknown":
            shadow = ""
            if (Path.cwd() / "mace").is_dir():
                shadow = ("\n  note: a mace/ subdirectory of the working directory shadows the editable "
                          "install -- cd into a repository root and retry")
            raise RuntimeError(
                "the installed mace at {} is not a git checkout: the Hessian loss needs the fork {} "
                "(pip uninstall -y mace-torch && bash <path>/openQHA-Hessian/install.sh, or "
                "pip install -e <path>/mace)".format(getattr(mace, "__file__", "?"), engine.MACE_FORK) + shadow)
        if info["mace_fork_dirty"]:
            raise RuntimeError(
                "the mace checkout at {} has uncommitted changes to tracked files; commit them so the "
                "run's Record names the code that produced it".format(info["mace_fork_path"]))
    return info


def split_files(dataset_dir, name, level, out_dir):
    """mace takes one file per split; the Dataset writes one file with a `split` key.
    Split it into `out_dir` and count the frames and the Hessians of each."""
    from ase.io import read, write
    from openqha.data import dataset as dataset_mod
    merged = dataset_mod.merged_file(dataset_dir, name, level)
    if not Path(merged).is_file():
        raise FileNotFoundError("no merged Dataset file at {}; run 04_dataset.py first".format(merged))
    frames = read(str(merged), index=":", format="extxyz")
    out, counts = {}, {}
    for split in ("train", "valid"):
        rows = [a for a in frames if a.info.get("split") == split]
        if not rows:
            raise ValueError("the Dataset has no {} frames in {}".format(split, merged))
        path = Path(out_dir) / "{}.{}.extxyz".format(split, level)
        write(str(path), rows, format="extxyz")
        out[split] = path
        counts[split] = (len(rows), sum(1 for a in rows if HESSIAN_KEY in a.info))
    return out, counts


_CONFIG_WEIGHT = re.compile(r'(?:^|\s)config_weight=("?)([-+0-9.eE]+)\1')


def replay_file_summary(path):
    """What the Record says about a Replay file, from its frame headers only (a
    68,000-frame file is read in seconds, never parsed as atoms): `n_frames`, the
    `config_weight` values seen (a set), and `config_weight` as one string -- the value,
    `mixed`, or `1.0` when no frame carries the key (mace's default)."""
    n, weights = 0, set()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            try:
                nat = int(s.split()[0])
            except ValueError:
                raise ValueError("{}: expected an atom count, read {!r}".format(path, s[:40]))
            header = fh.readline()
            m = _CONFIG_WEIGHT.search(header)
            weights.add(float(m.group(2)) if m else 1.0)
            n += 1
            for _ in range(nat):
                fh.readline()
    if not weights:
        label = "-"
    elif len(weights) == 1:
        label = repr(float(next(iter(weights))))
    else:
        label = "mixed"
    return dict(n_frames=n, weights=weights, config_weight=label)


def stage_two_weights(hessian_weight, forces_weight, swa_forces_weight, swa_hessian_weight=None):
    """The Stage Two weight rule: w_H^(2) = w_H x w_F^(2) / w_F unless given."""
    if swa_hessian_weight is not None:
        return float(swa_hessian_weight)
    if not forces_weight:
        return float(hessian_weight)
    return float(hessian_weight) * float(swa_forces_weight) / float(forces_weight)


def control_settings(max_epochs, lr=None, scheduler_patience=DEFAULT_SCHEDULER_PATIENCE, patience=DEFAULT_PATIENCE,
                     eval_interval=DEFAULT_EVAL_INTERVAL, ema=True, swa=True, start_swa=None, swa_lr=None,
                     swa_energy_weight=DEFAULT_SWA_ENERGY_WEIGHT, swa_forces_weight=DEFAULT_SWA_FORCES_WEIGHT,
                     hessian_weight=1.0, forces_weight=100.0, swa_hessian_weight=None):
    """Every control value with its default resolved, as one dict (the Record's and the
    argv's single source): LR, SCHEDULER_PATIENCE, PATIENCE, EVAL_INTERVAL, EMA, SWA,
    START_SWA (3 x max_epochs // 4, at least 1), SWA_LR (LR / 40), the Stage Two weights."""
    lr = DEFAULT_LR if lr is None else float(lr)
    max_epochs = int(max_epochs)
    return dict(
        LR=float(lr),
        SCHEDULER_PATIENCE=int(scheduler_patience), PATIENCE=int(patience), EVAL_INTERVAL=int(eval_interval),
        EMA=bool(ema), SWA=bool(swa),
        START_SWA=int(start_swa) if start_swa is not None else max(1, 3 * max_epochs // 4),
        SWA_LR=float(swa_lr) if swa_lr is not None else float(lr) / SWA_LR_RATIO,
        SWA_ENERGY_WEIGHT=float(swa_energy_weight), SWA_FORCES_WEIGHT=float(swa_forces_weight),
        SWA_HESSIAN_WEIGHT=stage_two_weights(hessian_weight, forces_weight, swa_forces_weight, swa_hessian_weight),
    )


def hessian_weight_balance(foundation_name, train_file, energy_weight=1.0, forces_weight=100.0,
                           probe="gaussian", n_probes=4, seed=123, device="cpu"):
    """The epoch-0 balance on the BASE model over `train_file` (`smoke_fit.epoch_zero_balance`):
    w_H = w_F L_F / L_H, with `L_H` measured under the run's probe setting (`probe`, `n_probes`,
    `seed`; `cartesian` is the exact full-matrix path). Refuses a train file without a
    Hessian frame (there is nothing to balance against)."""
    from . import smoke_fit
    calc, _name, _prov = engine.calculator(device=device, name=foundation_name)
    b = smoke_fit.epoch_zero_balance(calc, train_file, energy_weight=energy_weight, forces_weight=forces_weight,
                                     probe=probe, n_probes=n_probes, seed=seed)
    if not b["HESSIAN_WEIGHT_BALANCED"]:
        raise ValueError("no Hessian frame in {} (or L_H = 0): the balance rule has nothing to balance".format(train_file))
    return b


def _calculator_for(model, device="cpu"):
    """A calculator for `model`: a registered engine NAME goes through `engine.calculator`
    (its cache and the translation patch); a PATH -- a model this run just wrote, which no
    registry knows yet -- is loaded directly, with the same patch applied first, because the
    wrapper must be in place before any neighbour list is built."""
    if hasattr(model, "get_hessian"):
        return model
    p = Path(str(model))
    if not p.is_file():
        return engine.calculator(device=device, name=str(model))[0]
    from openqha.potentials import mace_patch
    mace_patch.apply(strict=True)
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=str(p), device=device, default_dtype=engine.DTYPE)


def exact_valid_hessian(model_name_or_path, valid_file, device="cpu"):
    """The EXACT Hessian term on the validation file: the frame-weighted mean of
    `||H_theta - H_r||_F^2 / (9 N^2)` over its labelled frames, from the full matrix
    (`get_hessian`, 3N HVPs per frame) -- the quantity the in-loop validation estimates
    with the fixed probes the Dataset stored.

    Measured twice per run: on the base model before training and on the
    fine-tuned model after it. Two readings of the same quantity on the same frames, so
    their difference is what the fine-tune moved on the target, free of the estimator's
    noise; the last epoch's probe reading beside them is what that noise cost. None when
    the file holds no Label. Costs one pass, not one per epoch: at 19 atoms a frame is
    3N = 57 force-graph passes, so ~900 validation frames are about a third of one epoch.
    """
    from . import smoke_fit
    calc = _calculator_for(model_name_or_path, device=device)
    b = smoke_fit.epoch_zero_balance(calc, valid_file, probe="cartesian")
    return None if not b["N_HESSIAN_FRAMES"] else float(b["L_H"])


def mace_argv(train_file, valid_file, run, work_dir, foundation, level, *, energy_weight=1.0,
              forces_weight=100.0, hessian_weight=1.0, probe="gaussian", n_probes=4,
              max_epochs=100, batch_size=4, valid_batch_size=None,
              seed=123, device="cpu", lr=None, multiheads=False, pt_train_file=None, pt_valid_file=None,
              scheduler_patience=DEFAULT_SCHEDULER_PATIENCE, patience=DEFAULT_PATIENCE,
              eval_interval=DEFAULT_EVAL_INTERVAL, ema=True, swa=True, start_swa=None, swa_lr=None,
              swa_energy_weight=DEFAULT_SWA_ENERGY_WEIGHT, swa_forces_weight=DEFAULT_SWA_FORCES_WEIGHT,
              swa_hessian_weight=None, extra=()):
    """mace's command line for one fine-tune, as a list. The keys are the Dataset's
    (`REF_*`), the loss is ours by name, the dtype is float64 because the Labels are,
    every control flag is explicit, and the Replay is the file and nothing
    else (no `--num_samples_pt`, the duplication threshold 0)."""
    ctl = control_settings(max_epochs, lr=lr, scheduler_patience=scheduler_patience, patience=patience,
                           eval_interval=eval_interval, ema=ema, swa=swa, start_swa=start_swa, swa_lr=swa_lr,
                           swa_energy_weight=swa_energy_weight, swa_forces_weight=swa_forces_weight,
                           hessian_weight=hessian_weight, forces_weight=forces_weight,
                           swa_hessian_weight=swa_hessian_weight)
    argv = [
        "--name", str(run),
        "--work_dir", str(work_dir),
        "--train_file", str(train_file),
        "--valid_file", str(valid_file),
        "--foundation_model", str(foundation),
        # the isolated-atom energies are the base model's: a fine-tune must not move the
        # energy zero, or the Labels' absolute energies stop meaning what they meant
        "--E0s", "foundation",
        "--energy_key", ENERGY_KEY, "--forces_key", FORCES_KEY, "--hessian_key", HESSIAN_KEY,
        "--loss", "external", "--loss_module", LOSS_MODULE,
        "--energy_weight", repr(float(energy_weight)),
        "--forces_weight", repr(float(forces_weight)),
        "--hessian_weight", repr(float(hessian_weight)),
        "--hessian_probe", str(probe),
        "--n_hessian_probes", str(int(n_probes)),
        "--max_num_epochs", str(int(max_epochs)),
        "--batch_size", str(int(batch_size)),
        "--valid_batch_size", str(int(valid_batch_size or batch_size)),
        "--seed", str(int(seed)),
        "--device", str(device),
        "--default_dtype", "float64",
        "--error_table", "PerAtomRMSE",
        "--save_cpu",
        # the control: explicit, so the config file and the Record say it
        "--lr", repr(ctl["LR"]),
        "--scheduler_patience", str(ctl["SCHEDULER_PATIENCE"]),
        "--patience", str(ctl["PATIENCE"]),
        "--eval_interval", str(ctl["EVAL_INTERVAL"]),
    ]
    if ctl["EMA"]:
        argv += ["--ema"]
    if ctl["SWA"]:
        argv += ["--swa", "--start_swa", str(ctl["START_SWA"]), "--swa_lr", repr(ctl["SWA_LR"]),
                 "--swa_energy_weight", repr(ctl["SWA_ENERGY_WEIGHT"]),
                 "--swa_forces_weight", repr(ctl["SWA_FORCES_WEIGHT"]),
                 "--swa_hessian_weight", repr(ctl["SWA_HESSIAN_WEIGHT"])]
    if multiheads:
        argv += ["--multiheads_finetuning", "True",
                 "--real_pt_data_ratio_threshold", repr(REAL_PT_DATA_RATIO_THRESHOLD)]
        if pt_train_file:
            argv += ["--pt_train_file", str(pt_train_file)]
        if pt_valid_file:
            argv += ["--pt_valid_file", str(pt_valid_file)]
    else:
        argv += ["--multiheads_finetuning", "False"]
    argv += [str(a) for a in extra]
    return argv


def argv_pairs(argv):
    """mace's command line as a mapping. A bare flag (`--save_cpu`) maps to True, so the
    config file and any reader of it see the same settings the parser did."""
    out, i = {}, 0
    argv = [str(a) for a in argv]
    while i < len(argv):
        key = argv[i].lstrip("-")
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[key] = argv[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


_EPOCH = re.compile(
    r"(?:Epoch (?P<epoch>\d+)|(?P<initial>Initial)):.*?(?:head: (?P<head>\w+), )?loss=(?P<loss>[-\d.eE+]+)"
    r"(?:.*?RMSE_E_per_atom=\s*(?P<rmse_e>[-\d.eE+]+) meV)?"
    r"(?:.*?RMSE_F=\s*(?P<rmse_f>[-\d.eE+]+) meV / A)?")
#: mace's pretraining head: its validation rows are not the fine-tune's curves
PT_HEAD = "pt_head"
INITIAL_EPOCH = -1
_VALID_TERMS = re.compile(
    r"openQHA loss \(valid\): energy=(?P<e>[-\d.eE+]+|-) forces=(?P<f>[-\d.eE+]+|-) hessian=(?P<h>[-\d.eE+]+|-)")
_STAGE_TWO = re.compile(r"Changing loss based on Stage Two Weights")
_HEAD = re.compile(r"Processing head (?P<head>\w+)")
_HEAD_COUNTS = re.compile(r"Total number of configurations: train=(?P<train>\d+), valid=(?P<valid>\d+)")
_PT_COUNTS = re.compile(r"Total number of configurations in pretraining: train=(?P<train>\d+), valid=(?P<valid>\d+)")


def _num(s):
    return None if s in (None, "-") else float(s)


def parse_epochs(log_text):
    """The epoch table out of mace's log. mace prints one line per validation (the
    `Initial:` one before any training is epoch -1); the loss's own line (`openQHA loss
    (valid): ...`, commit C) precedes it and carries the three terms, which are attached
    to the next Epoch line. The pretraining head's lines (`head: pt_head`) are dropped:
    they are not the fine-tune's curves. The training loss of the same epoch is in
    `results/<run>_*.txt` as JSON lines (`parse_results`)."""
    rows = []
    pending = None
    for line in log_text.splitlines():
        m = _VALID_TERMS.search(line)
        if m:
            pending = dict(valid_energy=_num(m.group("e")), valid_forces=_num(m.group("f")),
                           valid_hessian=_num(m.group("h")))
            continue
        m = _EPOCH.search(line)
        if m:
            if m.group("head") == PT_HEAD:
                pending = None
                continue
            row = dict(epoch=INITIAL_EPOCH if m.group("initial") else int(m.group("epoch")), split="valid",
                       loss=float(m.group("loss")),
                       rmse_e_per_atom_meV=float(m.group("rmse_e")) if m.group("rmse_e") else None,
                       rmse_f_meV_A=float(m.group("rmse_f")) if m.group("rmse_f") else None,
                       valid_energy=None, valid_forces=None, valid_hessian=None)
            if pending:
                row.update(pending)
                pending = None
            rows.append(row)
    return rows


def parse_stage_two_epoch(log_text):
    """The epoch mace switched to Stage Two at (the first Epoch line after the switch
    message), or -1."""
    switched = False
    for line in log_text.splitlines():
        if _STAGE_TWO.search(line):
            switched = True
            continue
        if switched:
            m = _EPOCH.search(line)
            if m:
                return int(m.group("epoch"))
    return -1


def parse_head_counts(log_text):
    """mace's per-head frame counts from its log: `{head: (train, valid)}` for every
    `Processing head` block, plus `pt` from the pretraining summary line (the count
    after any duplication; equal to the head's when the threshold is 0)."""
    out, head = {}, None
    for line in log_text.splitlines():
        m = _HEAD.search(line)
        if m:
            head = m.group("head")
            continue
        m = _HEAD_COUNTS.search(line)
        if m and head:
            out[head] = (int(m.group("train")), int(m.group("valid")))
            continue
        m = _PT_COUNTS.search(line)
        if m:
            out["pt"] = (int(m.group("train")), int(m.group("valid")))
    return out


def parse_results(results_dir):
    """mace's `results/*.txt`: one JSON object per evaluation (`mode`, `epoch`, `loss`,
    the error columns; the loss's three validation terms through commit C). Both splits,
    in the order they were written; the evaluation before any training (`epoch` null)
    is epoch -1; the pretraining head's rows (`head` = pt_head) are dropped."""
    rows = []
    for path in sorted(Path(results_dir).glob("*.txt")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("head") == PT_HEAD:
                continue
            valid = not str(r.get("mode", "")).startswith("opt")
            if r.get("epoch") is None:
                if not valid:
                    continue
                r["epoch"] = INITIAL_EPOCH
            rows.append(dict(epoch=int(r["epoch"]),
                             split="valid" if valid else "train",
                             loss=float(r["loss"]) if r.get("loss") is not None else None,
                             rmse_e_per_atom_meV=(float(r["rmse_e_per_atom"]) * 1000.0
                                                 if r.get("rmse_e_per_atom") is not None else None),
                             rmse_f_meV_A=(float(r["rmse_f"]) * 1000.0 if r.get("rmse_f") is not None else None),
                             valid_energy=_num(r.get("valid_energy_term")) if valid else None,
                             valid_forces=_num(r.get("valid_forces_term")) if valid else None,
                             valid_hessian=_num(r.get("valid_hessian_term")) if valid else None))
    return rows


def validation_curves(epochs):
    """The three curves as `{name: [(epoch, value), ...]}` from the epoch rows (valid
    rows with a value; one point per epoch, the last written)."""
    out = {}
    for name in ("valid_energy", "valid_forces", "valid_hessian"):
        by_epoch = {}
        for r in epochs:
            if r["split"] == "valid" and r.get(name) is not None:
                by_epoch[r["epoch"]] = float(r[name])
        out[name] = sorted(by_epoch.items())
    return out


def curve_moved(points, tol=FLAT_CURVE_TOL):
    """False when the curve has fewer than two points or its range is below `tol` of its
    magnitude: the Hessian term did not act (replaced in multihead mode, w_H 0, or a one-epoch run)."""
    vals = [v for _e, v in points]
    if len(vals) < 2:
        return False
    span = max(vals) - min(vals)
    scale = max(abs(v) for v in vals) or 1.0
    return span > tol * scale


def run_training(dataset_dir, tag, name, level, run, *, foundation=None, dry_run=False,
                 strict_fork=True, **settings):
    """One fine-tune, end to end: the run directory, the split files, mace's arguments,
    `mace.cli.run_train.run`, the Record. Returns the info dict (`--dry-run`: the argv
    and the info without training)."""
    if "num_samples_pt" in settings:
        raise TypeError("num_samples_pt is not a setting: the Replay's size is its file's; "
                        "draw it with scripts/tooling/s0_spice_pt_draw.py --n N")
    dataset_dir = Path(dataset_dir)
    run_dir = dataset_dir / STEP / run
    run_dir.mkdir(parents=True, exist_ok=True)
    fork = check_fork(strict=strict_fork)

    foundation_name = foundation or engine.engine_name()
    foundation_path = engine.model_path(foundation_name)

    files, counts = split_files(dataset_dir, name, level, run_dir)
    settings = dict(settings)
    # not mace's: the two full-matrix readings of the Hessian term on the validation file
    exact_anchors = bool(settings.pop("exact_anchors", True))
    rule, balance = "given", dict(L_E=0.0, L_F=0.0, L_H=0.0)
    if str(settings.get("hessian_weight", 1.0)) == "balance":
        rule = "balance"
        balance = hessian_weight_balance(foundation_name, files["train"], settings.get("energy_weight", 1.0),
                                         settings.get("forces_weight", 100.0),
                                         probe=settings.get("probe", "gaussian"),
                                         n_probes=settings.get("n_probes", 4),
                                         seed=settings.get("seed", 123),
                                         device=settings.get("device", "cpu"))
        settings["hessian_weight"] = balance["HESSIAN_WEIGHT_BALANCED"]
    argv = mace_argv(files["train"], files["valid"], run, run_dir, foundation_path, level, **settings)
    config_file = run_dir / "config.yaml"
    config_file.write_text("\n".join("{}: {}".format(k, v) for k, v in argv_pairs(argv).items()) + "\n",
                           encoding="utf-8")
    ctl = control_settings(settings.get("max_epochs", 100), lr=settings.get("lr"),
                           scheduler_patience=settings.get("scheduler_patience", DEFAULT_SCHEDULER_PATIENCE),
                           patience=settings.get("patience", DEFAULT_PATIENCE),
                           eval_interval=settings.get("eval_interval", DEFAULT_EVAL_INTERVAL),
                           ema=settings.get("ema", True), swa=settings.get("swa", True),
                           start_swa=settings.get("start_swa"), swa_lr=settings.get("swa_lr"),
                           swa_energy_weight=settings.get("swa_energy_weight", DEFAULT_SWA_ENERGY_WEIGHT),
                           swa_forces_weight=settings.get("swa_forces_weight", DEFAULT_SWA_FORCES_WEIGHT),
                           hessian_weight=settings.get("hessian_weight", 1.0),
                           forces_weight=settings.get("forces_weight", 100.0),
                           swa_hessian_weight=settings.get("swa_hessian_weight"))

    multiheads = bool(settings.get("multiheads", False))
    pt_file = settings.get("pt_train_file") if multiheads else None
    replay = replay_file_summary(pt_file) if pt_file else dict(n_frames=0, config_weight="-")
    n_hess = counts["train"][1]

    identity = package_identity()
    info = dict(
        RUN=str(run), TAG=str(tag), NAME=str(name), LEVEL=str(level),
        DATASET_DIR=str(dataset_dir), INDEX_FILE=str(dataset_dir / "index.dat"),
        TRAIN_FILE=str(files["train"]), VALID_FILE=str(files["valid"]),
        N_TRAIN=counts["train"][0], N_TRAIN_HESSIAN=n_hess,
        N_VALID=counts["valid"][0], N_VALID_HESSIAN=counts["valid"][1],
        FOUNDATION_MODEL=foundation_name, FOUNDATION_FILE=str(foundation_path),
        CONFIG_FILE=str(config_file), CONFIG_SHA256=sha256_file(config_file),
        LOSS=LOSS_MODULE,
        ENERGY_WEIGHT=float(settings.get("energy_weight", 1.0)),
        FORCES_WEIGHT=float(settings.get("forces_weight", 100.0)),
        HESSIAN_WEIGHT=float(settings.get("hessian_weight", 1.0)),
        HESSIAN_WEIGHT_RULE=rule,
        BALANCE_L_E=float(balance["L_E"] or 0.0), BALANCE_L_F=float(balance["L_F"] or 0.0),
        BALANCE_L_H=float(balance["L_H"] or 0.0),
        BALANCE_PROBE=str(balance.get("PROBE", "-")), BALANCE_N_PROBES=int(balance.get("N_PROBES", 0)),
        PROBE=str(settings.get("probe", "gaussian")),
        N_PROBES=int(settings.get("n_probes", 4)),
        VALID_PROBES=phl_loss.VALID_PROBES_LABEL,
        MAX_NUM_EPOCHS=int(settings.get("max_epochs", 100)),
        BATCH_SIZE=int(settings.get("batch_size", 4)),
        SEED=int(settings.get("seed", 123)),
        DEVICE=str(settings.get("device", "cpu")), DTYPE="float64",
        STAGE_TWO_EPOCH=-1,
        MULTIHEADS=multiheads,
        PT_TRAIN_FILE=str(pt_file or "-"),
        PT_VALID_FILE=str(settings.get("pt_valid_file") or "-") if multiheads else "-",
        PT_N_FRAMES=int(replay["n_frames"]),
        PT_CONFIG_WEIGHT=str(replay["config_weight"]),
        PT_HEAD_TRAIN=-1, PT_HEAD_VALID=-1, FT_HEAD_TRAIN=-1, FT_HEAD_VALID=-1,
        REPLAY_PER_HESSIAN_FRAME=(float(replay["n_frames"]) / n_hess) if (n_hess and replay["n_frames"]) else 0.0,
        REAL_PT_DATA_RATIO_THRESHOLD=REAL_PT_DATA_RATIO_THRESHOLD if multiheads else 0.0,
        HESSIAN_CURVE_MOVED=False,
        VALID_HESSIAN_EXACT_BEFORE=-1.0, VALID_HESSIAN_EXACT_AFTER=-1.0,
        VALID_HESSIAN_PROBE_LAST=-1.0, VALID_PROBE_OFFSET_RUN=-1.0,
        EXACT_ANCHORS=exact_anchors,
        MACE_VERSION=fork["mace_version"], MACE_FORK=engine.MACE_FORK,
        MACE_FORK_COMMIT=fork["mace_fork_commit"],
        HL_PACKAGE_VERSION=identity["HL_PACKAGE_VERSION"], HL_PACKAGE_COMMIT=identity["HL_PACKAGE_COMMIT"],
    )
    info.update(ctl)
    if dry_run:
        return dict(info=info, argv=argv, run_dir=run_dir, epochs=[], dry_run=True)

    if info["EXACT_ANCHORS"]:
        before = exact_valid_hessian(foundation_name, files["valid"], settings.get("device", "cpu"))
        if before is not None:
            info["VALID_HESSIAN_EXACT_BEFORE"] = before

    t0 = time.time()
    _run_mace(argv, run_dir)
    seconds = time.time() - t0

    model_file = run_dir / "{}.model".format(run)
    if not model_file.is_file():
        stage_two = run_dir / "{}_stagetwo.model".format(run)
        model_file = stage_two if stage_two.is_file() else model_file
    log_text = "\n".join(p.read_text(errors="replace") for p in sorted((run_dir / "logs").glob("*.log")))
    epochs = parse_results(run_dir / "results")
    if not epochs:
        epochs = parse_epochs(log_text)
    heads = parse_head_counts(log_text)
    if "pt" in heads:
        info["PT_HEAD_TRAIN"], info["PT_HEAD_VALID"] = heads["pt"]
    elif "pt_head" in heads:
        info["PT_HEAD_TRAIN"], info["PT_HEAD_VALID"] = heads["pt_head"]
    ft = [h for h in heads if h not in ("pt", "pt_head")]
    if ft:
        info["FT_HEAD_TRAIN"], info["FT_HEAD_VALID"] = heads[ft[0]]
    info["STAGE_TWO_EPOCH"] = parse_stage_two_epoch(log_text)
    info["HESSIAN_CURVE_MOVED"] = curve_moved(validation_curves(epochs)["valid_hessian"])
    info["N_EPOCHS"] = len({r["epoch"] for r in epochs if r["epoch"] != INITIAL_EPOCH})
    info["SECONDS"] = float(seconds)
    info["SECONDS_PER_EPOCH"] = float(seconds / max(1, info["N_EPOCHS"]))
    info["MODEL_FILE"] = str(model_file) if model_file.is_file() else "-"
    if info["EXACT_ANCHORS"] and model_file.is_file():
        after = exact_valid_hessian(str(model_file), files["valid"], settings.get("device", "cpu"))
        if after is not None:
            info["VALID_HESSIAN_EXACT_AFTER"] = after
            probe_last = validation_curves(epochs)["valid_hessian"]
            last = next((v for _e, v in reversed(probe_last) if v is not None), None) if probe_last else None
            if last is not None:
                info["VALID_HESSIAN_PROBE_LAST"] = float(last)
                if after > 0:
                    info["VALID_PROBE_OFFSET_RUN"] = float(abs(float(last) - after) / after)
    write_record(run_dir, info, epochs)
    return dict(info=info, argv=argv, run_dir=run_dir, epochs=epochs, dry_run=False)


def _run_mace(argv, run_dir):
    """`mace.cli.run_train.run(args)` with our argv, from the run directory. mace parses
    `sys.argv`, so it is swapped for the call and restored after it."""
    from mace.cli.run_train import run as mace_run
    from mace.tools import build_default_arg_parser
    args = build_default_arg_parser().parse_args(argv)
    cwd = os.getcwd()
    old_argv = sys.argv
    try:
        os.chdir(run_dir)
        sys.argv = ["mace_run_train"] + list(argv)
        mace_run(args)
    finally:
        sys.argv = old_argv
        os.chdir(cwd)


def _epoch_row(r):
    return dict(epoch=r["epoch"], split=r["split"], loss=r.get("loss"),
                rmse_e_per_atom_meV=r.get("rmse_e_per_atom_meV"), rmse_f_meV_A=r.get("rmse_f_meV_A"),
                valid_energy=r.get("valid_energy"), valid_forces=r.get("valid_forces"),
                valid_hessian=r.get("valid_hessian"))


def write_record(run_dir, info, epochs):
    epochs = [_epoch_row(r) for r in epochs]
    rows = [dict(EPOCH=r["epoch"], SPLIT=r["split"], LOSS=r["loss"],
                 RMSE_E_PER_ATOM_MEV=r["rmse_e_per_atom_meV"], RMSE_F_MEV_A=r["rmse_f_meV_A"],
                 VALID_ENERGY=r["valid_energy"], VALID_FORCES=r["valid_forces"], VALID_HESSIAN=r["valid_hessian"])
            for r in epochs]
    missing = prop.write(Path(run_dir) / (STEP + ".toml"), {"Calculation_Info": info, "Epoch": rows},
                         SCHEMA, prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("train.toml keys outside the schema: {}".format(missing))
    dat.write_table(Path(run_dir) / (STEP + ".dat"), epochs, list(EPOCH_ROW), EPOCH_ROW)
    _write_report(Path(run_dir) / (STEP + ".out"), info, epochs)


def _fmt(v, spec="{:.6e}"):
    return "-" if v is None else spec.format(v)


def _record_num(info, key, default=-1.0):
    """`info[key]` as a float, or `default` when it is absent or not a number: a report must
    print what a Record holds, not raise on it."""
    try:
        return float(info.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _write_report(path, info, epochs):
    rep = report.Report(PROGNAME, "Fine-tune {!r} of {} on {}".format(
        info["RUN"], info["FOUNDATION_MODEL"], info["NAME"]))
    rep.section("the Dataset")
    for k in ("TAG", "NAME", "LEVEL", "DATASET_DIR", "TRAIN_FILE", "N_TRAIN", "N_TRAIN_HESSIAN",
              "VALID_FILE", "N_VALID", "N_VALID_HESSIAN"):
        rep.kv(k, info.get(k))
    rep.section("the loss (eq. 11; the Cartesian target; T04 / T05)")
    if info.get("EXACT_ANCHORS") is True and _record_num(info, "VALID_HESSIAN_EXACT_BEFORE") >= 0:
        b_, a_ = _record_num(info, "VALID_HESSIAN_EXACT_BEFORE"), _record_num(info, "VALID_HESSIAN_EXACT_AFTER")
        rep.note("EXACT ANCHORS: the full-matrix Hessian term on the validation file, base model "
                 "{:.4e}{} -- the same quantity the in-loop validation estimates with {} fixed probes per frame. The "
                 "pair is free of estimator noise; the last epoch's probe reading is {} and its relative distance from "
                 "the exact value is {}. Neither number is a generalisation reading: the validation frames belong to "
                 "TRAINING molecules; the judge's test split is where generalisation is read.".format(
                     b_, "" if a_ < 0 else " -> fine-tuned {:.4e} ({:+.1%})".format(a_, a_ / b_ - 1.0 if b_ else 0.0),
                     phl_loss.VALID_N_PROBES,
                     "-" if _record_num(info, "VALID_HESSIAN_PROBE_LAST") < 0 else "{:.4e}".format(_record_num(info, "VALID_HESSIAN_PROBE_LAST")),
                     "-" if _record_num(info, "VALID_PROBE_OFFSET_RUN") < 0 else "{:.1%}".format(_record_num(info, "VALID_PROBE_OFFSET_RUN"))))
    for k in ("LOSS", "ENERGY_WEIGHT", "FORCES_WEIGHT", "HESSIAN_WEIGHT", "HESSIAN_WEIGHT_RULE",
              "BALANCE_L_E", "BALANCE_L_F", "BALANCE_L_H", "BALANCE_PROBE", "BALANCE_N_PROBES",
              "PROBE", "N_PROBES", "VALID_PROBES"):
        rep.kv(k, info.get(k))
    if info.get("HESSIAN_WEIGHT_RULE") == "balance":
        rep.note("HESSIAN_WEIGHT = FORCES_WEIGHT x BALANCE_L_F / BALANCE_L_H: the Hessian term enters the epoch-0 gradient "
                 "with the force term's share on the base model (T05, section 4); BALANCE_L_H was measured with the run's "
                 "probe setting ({} k={}) -- the exact full-matrix reading is the anchors' and the judge's job.".format(
                     info.get("BALANCE_PROBE"), info.get("BALANCE_N_PROBES")))
    rep.section("the Replay")
    for k in ("MULTIHEADS", "PT_TRAIN_FILE", "PT_VALID_FILE", "PT_N_FRAMES", "PT_CONFIG_WEIGHT", "PT_HEAD_TRAIN",
              "PT_HEAD_VALID", "FT_HEAD_TRAIN", "FT_HEAD_VALID", "REPLAY_PER_HESSIAN_FRAME",
              "REAL_PT_DATA_RATIO_THRESHOLD"):
        rep.kv(k, info.get(k))
    if info.get("MULTIHEADS"):
        rep.note("REPLAY_PER_HESSIAN_FRAME = PT_N_FRAMES / N_TRAIN_HESSIAN is the Replay's size as the S0 scan rows "
                 "R0-R4 define it (R1 ~ 0.3, R2 ~ 1, R3 ~ 4 on draw300); the same file is a different ratio on "
                 "every Dataset. The counts PT_HEAD_* / FT_HEAD_* are mace's own, parsed from its log, and equal "
                 "the files' because the duplication threshold is 0.")
    rep.section("the control (the base's recipe)")
    for k in ("LR", "SCHEDULER_PATIENCE", "PATIENCE", "EVAL_INTERVAL", "EMA", "SWA", "START_SWA", "SWA_LR",
              "STAGE_TWO_EPOCH"):
        rep.kv(k, info.get(k))
    rep.section("StageTwo weights (w_H^(2) = w_H x w_F^(2) / w_F)")
    for k in ("SWA_ENERGY_WEIGHT", "SWA_FORCES_WEIGHT", "SWA_HESSIAN_WEIGHT"):
        rep.kv(k, info.get(k))
    rep.section("identity")
    for k in ("FOUNDATION_MODEL", "FOUNDATION_FILE", "MODEL_FILE", "CONFIG_FILE", "CONFIG_SHA256",
              "MACE_VERSION", "MACE_FORK", "MACE_FORK_COMMIT", "HL_PACKAGE_VERSION", "HL_PACKAGE_COMMIT",
              "SEED", "DEVICE", "DTYPE"):
        rep.kv(k, info.get(k))
    rep.section("cost")
    for k in ("MAX_NUM_EPOCHS", "N_EPOCHS", "BATCH_SIZE", "SECONDS", "SECONDS_PER_EPOCH"):
        rep.kv(k, info.get(k))
    if epochs:
        curves = validation_curves(epochs)
        rep.section("validation curves (the three terms per epoch; the Hessian term is the fixed-probe estimator)")
        rep.kv("HESSIAN_CURVE_MOVED", info.get("HESSIAN_CURVE_MOVED"))
        if not info.get("HESSIAN_CURVE_MOVED"):
            rep.note("WARNING: the Hessian term did not move across the validation epochs -- the term did not act "
                     "(the loss replaced in multihead mode, or w_H 0) or the run is one epoch.")
        rep.table(["epoch", "split", "loss", "RMSE E/atom meV", "RMSE F meV/A", "valid E term", "valid F term",
                   "valid H term"],
                  [[r["epoch"], r["split"], _fmt(r["loss"], "{:.6f}"),
                    _fmt(r["rmse_e_per_atom_meV"], "{:.2f}"), _fmt(r["rmse_f_meV_A"], "{:.2f}"),
                    _fmt(r["valid_energy"]), _fmt(r["valid_forces"]), _fmt(r["valid_hessian"])]
                   for r in epochs])
        del curves
    rep.write(path)


def registry_entry(info, stamp=None):
    """The `ENGINES` lines for the fine-tuned potential: the
    Dataset's index and the config SHA are its `source`, so the numbers can be traced.
    One fixed revision per run, never a moving pointer: both the key and
    the file name carry the UTC stamp, and the file lives under `mace_off23_<campaign>/`
    as `<run>+<YYYYMMDD-HHMMSS>.model`."""
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = "{}-{}+{}".format(info["TAG"], info["RUN"], stamp)
    filename = "mace_off23_{}/{}+{}.model".format(info["TAG"], info["RUN"], stamp)
    replay = ""
    if info.get("MULTIHEADS") and info.get("PT_N_FRAMES"):
        replay = "; Replay {} frames ({:.2f} per Hessian frame, config_weight {})".format(
            info["PT_N_FRAMES"], float(info.get("REPLAY_PER_HESSIAN_FRAME") or 0.0), info.get("PT_CONFIG_WEIGHT"))
    return dict(name=name,
                filename=filename,
                source="{} + config {}".format(info["INDEX_FILE"], info["CONFIG_SHA256"][:16]),
                note="Fine-tuned on {} ({} frames, {} with Hessians) with the Hessian loss "
                     "(the full Cartesian matrix, w_H {}, probe {} k={}){}; mace fork {}.".format(
                         info["NAME"], info["N_TRAIN"], info["N_TRAIN_HESSIAN"],
                         info["HESSIAN_WEIGHT"], info["PROBE"], info["N_PROBES"], replay,
                         info["MACE_FORK_COMMIT"][:12]))
