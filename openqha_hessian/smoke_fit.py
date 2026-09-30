"""The smoke fit: what the campaign's settings are measured from.

PRODUCTION, but its numbers are NOT a claim about generalisation. A fit Dataset drops the
pinned rule and splits the same molecules' frames by frame, so every number out of it is
interpolation within those molecules (`PURPOSE = fit` in the Record, and the judge says so
on every row). What it is for:

  the balance     `w_H` such that `w_H L_H = w_F L_F` on the BASE model at epoch 0. The
                  Cartesian 0.25-0.30 band of PHL / Rodriguez does not transfer under mass
                  weighting, so the weight is measured, not quoted.
  the cost        seconds per epoch for each probe setting, against the E/F loss: the
                  `(2 + 2k)` against `(2 + 6N)` cost claim, in seconds.
  the ceiling     what the exact loss (the deterministic `3N` probes) reaches, so the
                  sampled runs are read against something.
  the replay      how many SPICE frames per Hessian-labelled frame a `--num_samples_pt`
                  buys HERE -- the number the campaign's choice is scaled from. The same
                  flag means 152 on the smoke set and 0.17 on the campaign, so a replay
                  setting quoted without this ratio says nothing.

Nothing here decides anything: it measures, writes the numbers down, and the settings
stay the user's call.
"""
import time
from pathlib import Path

import numpy as np

from openqha.data import dataset as dataset_mod
from openqha.store import property as prop
from openqha_hessian import phl
from . import run as train_run

PROGNAME = "openQHA hl_smoke_fit"

#: the probe settings the cost table walks, in the order it reports them
COST_SETTINGS = (
    dict(label="energy_forces", probe=None),
    dict(label="gaussian k=2", probe="gaussian", n_probes=2),
    dict(label="gaussian k=4", probe="gaussian", n_probes=4),
    dict(label="gaussian k=8", probe="gaussian", n_probes=8),
    dict(label="cartesian (exact)", probe="cartesian"),
)


def labelled_frames(path, hessian_key="REF_hessian"):
    """The frames of an extxyz that carry a reference Hessian, and the rest."""
    from ase.io import read
    frames = read(str(path), index=":", format="extxyz")
    with_h = [a for a in frames if hessian_key in a.info or a.info.get("hessian") is not None]
    return frames, with_h


def build_fit_dataset(source_dir, level, out_dir, name, seed=0, valid_fraction=0.2, splits=("test", "train", "valid")):
    """A FIT Dataset from the labelled frames of an existing one: the same frames, split
    by frame, the pinned rule off. Every number from it is interpolation within these
    molecules; the Record says `PURPOSE = fit` so that no table quotes it otherwise.

    This does not run `01_select` / `02_frames`: it re-splits frames that are already
    labelled, which is what a cost and weight measurement needs and all it needs.
    """
    from ase.io import read
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in splits:
        p = Path(source_dir) / "{}.{}.extxyz".format(split, level)
        if p.is_file():
            frames.extend(read(str(p), index=":", format="extxyz"))
    if not frames:
        raise FileNotFoundError("no labelled frame in {} for splits {} at level {}".format(
            source_dir, ", ".join(splits), level))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frames))
    n_valid = max(1, int(round(valid_fraction * len(frames)))) if len(frames) > 1 else 0
    valid_idx = set(order[:n_valid].tolist())
    rows = [(a, "valid" if i in valid_idx else "train") for i, a in enumerate(frames)]
    per_split = {"train": [r for r in rows if r[1] == "train"], "valid": [r for r in rows if r[1] == "valid"]}
    for split, sel in per_split.items():
        if sel:
            dataset_mod._write_split(out_dir / "{}.{}.extxyz".format(split, level), sel, reference=True)
    dataset_mod._write_split(dataset_mod.merged_file(out_dir, name, level), rows, reference=True)
    n_h = sum(1 for a, _s in rows if a.info.get("hessian") is not None)
    info = dict(NAME=str(name), PURPOSE="fit", LEVEL=str(level), SOURCE=str(source_dir), SEED=int(seed),
                N_FRAMES=len(rows), N_HESSIAN_FRAMES=int(n_h),
                N_TRAIN=len(per_split["train"]), N_VALID=len(per_split["valid"]),
                N_MOLECULES=len({a.info.get("qm9_index") for a, _s in rows}))
    prop.write(out_dir / "fit_dataset.toml", {"Calculation_Info": info},
               {"Calculation_Info": {k: ("String" if isinstance(v, str) else "Integer", None,
                                         "see openqha_hessian.smoke_fit.build_fit_dataset") for k, v in info.items()}},
               prop.NORMAL_TERMINATION, PROGNAME)
    return out_dir, info


def replay_ratio(n_train, n_hessian, num_samples_pt):
    """What a `--num_samples_pt` actually buys, in frames.

    mace concatenates the two heads' training sets and shuffles, so every frame of either
    head is seen once per epoch: the ratio is set by dataset size, NOT by a step loop as
    in PFT's algorithm 1 (K = 4 upstream steps per Hessian step). `per_hessian_frame` is
    the number to compare with PFT's 4.
    """
    n_train, n_hessian, n_pt = int(n_train), int(n_hessian), int(num_samples_pt)
    return dict(N_TRAIN=n_train, N_HESSIAN=n_hessian, NUM_SAMPLES_PT=n_pt,
                REPLAY_PER_TRAIN=(n_pt / n_train) if n_train else None,
                REPLAY_PER_HESSIAN_FRAME=(n_pt / n_hessian) if n_hessian else None,
                PFT_REFERENCE=4.0)


def epoch_zero_balance(calc, train_file, energy_weight=1.0, forces_weight=100.0, probe="cartesian",
                       hessian_key="REF_hessian", energy_key="REF_energy",
                       forces_key="REF_forces"):
    """The three terms of eq. 11 on the BASE model, before a single step.

    `L_E` and `L_F` are mace's per-config-weighted squared errors as the loss computes
    them (energy per atom, forces per component); `L_H` is the target EXACTLY -- the full
    matrix, eq. 1', the one target -- averaged over the frames that carry a
    Label.
    Returns the terms and `w_H = w_F L_F / L_H`, the weight at which the Hessian term
    enters with the same gradient share as the forces at epoch 0.
    """
    from ase.io import read
    frames = read(str(train_file), index=":", format="extxyz")
    e_sq, f_sq, h_vals = [], [], []
    for atoms in frames:
        at = atoms.copy()
        at.calc = calc
        e_ref = atoms.info.get(energy_key, atoms.info.get("energy"))
        f_ref = atoms.arrays.get(forces_key)
        if f_ref is None:
            try:
                f_ref = atoms.get_forces()
            except Exception:                                     # noqa: BLE001
                f_ref = None
        if e_ref is not None:
            e_sq.append(((float(at.get_potential_energy()) - float(e_ref)) / len(at)) ** 2)
        if f_ref is not None:
            f_sq.append(np.mean((np.asarray(at.get_forces()) - np.asarray(f_ref)) ** 2))
        flat = atoms.info.get(hessian_key, atoms.info.get("hessian"))
        if flat is None:
            continue
        n3 = 3 * len(atoms)
        h_r = np.asarray(flat, dtype=float).reshape(n3, n3)
        from .judge import hessian_at
        h_e = hessian_at(calc, atoms)
        h_vals.append(phl.loss_full(h_e, h_r))
    l_e = float(np.mean(e_sq)) if e_sq else None
    l_f = float(np.mean(f_sq)) if f_sq else None
    l_h = float(np.mean(h_vals)) if h_vals else None
    out = dict(N_FRAMES=len(frames), N_HESSIAN_FRAMES=len(h_vals), PROBE=probe,
               L_E=l_e, L_F=l_f, L_H=l_h, ENERGY_WEIGHT=float(energy_weight), FORCES_WEIGHT=float(forces_weight),
               WE_LE=None if l_e is None else energy_weight * l_e,
               WF_LF=None if l_f is None else forces_weight * l_f,
               HESSIAN_WEIGHT_BALANCED=None)
    if l_h and l_f:
        out["HESSIAN_WEIGHT_BALANCED"] = float(forces_weight * l_f / l_h)
    return out


def cost_table(dataset_dir, tag, name, level, settings=COST_SETTINGS, epochs=1, batch_size=2,
               run_prefix="cost", strict_fork=False, **common):
    """Seconds per epoch for each probe setting, and the ratio to the E/F loss. One epoch
    each: what is being measured is the cost of a step, not convergence."""
    rows = []
    ef_seconds = None
    for s in settings:
        label = s["label"]
        extra = dict(common)
        if s["probe"] is None:
            extra["hessian_weight"] = 0.0
            probe, n_probes = "gaussian", 1                # the term is weighted to zero
        else:
            probe, n_probes = s["probe"], s.get("n_probes", 4)
        t0 = time.time()
        out = train_run.run_training(dataset_dir, tag, name, level,
                                     "{}_{}".format(run_prefix, label.split()[0] + str(n_probes)),
                                     probe=probe, n_probes=n_probes, max_epochs=epochs,
                                     batch_size=batch_size, strict_fork=strict_fork, **extra)
        seconds = out["info"]["SECONDS"]
        per_epoch = out["info"]["SECONDS_PER_EPOCH"]
        if s["probe"] is None:
            ef_seconds = per_epoch
        rows.append(dict(SETTING=label, PROBE=probe if s["probe"] else "-", N_PROBES=n_probes if s["probe"] else 0,
                         SECONDS=float(seconds), SECONDS_PER_EPOCH=float(per_epoch),
                         RATIO_TO_EF=None if not ef_seconds else float(per_epoch / ef_seconds),
                         RUN=out["info"]["RUN"], N_TRAIN=out["info"]["N_TRAIN"],
                         N_TRAIN_HESSIAN=out["info"]["N_TRAIN_HESSIAN"]))
        del t0
    return rows
