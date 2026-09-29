"""The ruler: what a fine-tuned potential is judged by (Algorithm 5 of spec-phl-verbatim.md).

PRODUCTION. Ticket 14 of the Hessian-learning set.

The judge reads only SHIPPED paths: the full Cartesian Hessian from
`MACECalculator.get_hessian` (mace's `compute_hessians_vmap`, no graph) against the
Label, through `hessian_compare` -- the same four metric families every earlier
comparison in this repository used. **Nothing here calls the estimator or the training
loss.** The optimiser reads eq. 6; the judge reads the Cartesian target
`||H - H_r||_F^2 / (9 N^2)`, exactly, from the full matrix.
That separation is the point: a loss that flatters itself cannot flatter the ruler.

What one judge run reports:

  per frame        `hessian_compare`'s metrics of the engine and of the base model at the
                   same geometry, plus the training target `||H - H_r||_F^2 / (9 N^2)`
                   exactly (the gate's quantity, from `phl.loss_full`)
  per class        the structure classes of `index.dat`, so "how are we doing on
                   epoxides" has an answer
  per distribution `out_of_molecule` (whole molecules the fine-tune never saw: the pinned
                   seven and, under the by-molecule split of S0-C-65, every drawn test
                   molecule -- so this is the generalisation reading the gate row rests on),
                   `interpolation` (a test frame of a TRAINING molecule: only the by-frame
                   split of the smoke / fit Datasets produces one; empty in production)
                   and `in_distribution` (the shipped molecules, which MACE-OFF23 did see)
  anharmonic       reference modes the entropy tier must not be judged on (round-2 Q6 (a),
                   assumed): omega_r < ANHARMONIC_CM, or a `mode_curvature` self-check
                   above FD_SELF_CHECK_CM where a profile exists
  thermochemistry  read from the msRRHO Records on disk, never recomputed here (running
                   CREST and an optimisation inside the judge would make the judge a
                   producer of the numbers it judges)
  forgetting       E and F RMSE on a fixed SPICE draw, engine against base (round-2 Q7)
  displacement     the held-out generator's labelled frames (S0-C-54: displaced, merged,
                   saddle -- never trained on) binned by RMS displacement from the basin
                   (0 / < 0.08 / < 0.15 / >= 0.15 A), per distribution: H where a Label
                   Hessian exists, E and F everywhere -- Rodriguez's extrapolation readout
  MD ramp          Rodriguez's stability protocol on the pinned molecules: from each
                   model's own minimum, Langevin from 5 K, +5 K every 5 ps, until an atom
                   pair's 50-step mean distance leaves [0.75, 1.5] x its equilibrium value;
                   the failure temperature and time, engine beside base
  verdict          one line per row (round-2 Q8 as the spec assumes; ticket 22, S0-C-58/59):
                   GATE rows decide -- the HESSIAN MATRIX itself against the Label on the
                   held-out Hessian frames, ||H_theta - H_r||_F^2 / (9 N^2), the training
                   target's own number, engine against base (no worse); the
                   in_distribution no-degradation; the forgetting line. Everything
                   computed FROM the matrix afterwards is post-processing and a REFERENCE
                   row, reported with PASS / FAIL against its number and never moving the
                   verdict: the low-mode frequency line (the standard vibrational analysis
                   of the trained matrix -- mass weighting + Eckart projection -- never the
                   training loss), the msRRHO entropy at the
                   engine's own minima, the held-out generator's H against the base (the
                   RMS bins), the MD ramp

Falsifiability. `--engine base` must PASS every gate row and show exactly 0 against
itself; `ScaledCalculator(calc, 0.9)` (forces x0.9, Hessian x0.81) must FAIL the Hessian
gate and with it the verdict. A judge that cannot fail is not a judge.
"""
import math
from pathlib import Path

import numpy as np

from openqha.data import dataset as dataset_mod
from openqha.store import dat, layout, property as prop, report
from openqha.thermochem import hessian as hessian_mod
from openqha.thermochem import hessian_compare as hc
from openqha_hessian import package_identity
from openqha_hessian import phl

PROGNAME = "openQHA hl_judge"
STEP = "judge"

#: a reference mode below this is set aside from the entropy tier (round-2 Q6 (a))
ANHARMONIC_CM = 30.0
#: ... and so is one whose along-mode finite-difference self-check exceeds this
FD_SELF_CHECK_CM = 5.0
#: "low mode" of the judge's headline number, as everywhere else in this repository
LOW_CM = hc.LOW_CM

#: the thresholds of round-2 Q8 as the spec assumes them; `judge.run(thresholds=)` overrides
THRESHOLDS = dict(
    hessian_cartesian_vs_base=0.0,  # engine's ||dH||^2/(9N^2) on the held-out Hessian frames <= the base's (ratio - 1 <= 0)
    low_mode_mae_cm=8.5,            # the in-distribution value (propanal, S0-C-41); a reference row since S0-C-59
    model_error_s_ref=0.2,          # cal/mol/K, excluding the anharmonic modes
    in_distribution_degradation=0.15,   # no HIP metric worse than the base by more than this
    forgetting=0.15,                # SPICE E/F RMSE within this of the base's
)

#: the molecules MACE-OFF23 was trained on among the pinned seven (S0-C-40)
IN_DISTRIBUTION = ("dsgdb9nsd_000018", "dsgdb9nsd_000019", "dsgdb9nsd_000035", "dsgdb9nsd_000036")

#: the HIP metrics the no-degradation line watches
HIP_METRICS = ("HESSIAN_MAE", "FREQ_MAE_CM", "FREQ_MAE_LOW_CM", "EIGVAL_MAE_ECKART")

#: the RMS-displacement bins of the reference rows (upper edge exclusive, A); a basin
#: frame (rms 0) is its own bin
RMS_BINS = ((0.0, "0"), (0.08, "<0.08"), (0.15, "<0.15"), (float("inf"), ">=0.15"))
#: the verdict's rows: which decide (gate: the Hessian MATRIX against the Label, engine vs
#: base; no degradation in distribution; forgetting) and which are computed from the matrix
#: afterwards -- post-processing, reported only (reference; S0-C-58/59)
GATE_ROWS = ("held_out_hessian_cartesian", "in_distribution_degradation", "forgetting")
REFERENCE_ROWS = ("held_out_low_mode_mae_cm", "model_error_s_ref_cal_per_mol_K", "held_out_generator_hessian_vs_base", "md_ramp_K")
#: the MD ramp (Rodriguez 2025's protocol): start, step, hold per step, ceiling, timestep
RAMP = dict(start_K=5.0, step_K=5.0, step_ps=5.0, max_K=600.0, timestep_fs=1.0, window_steps=50,
            upper=1.5, lower=0.75, friction_per_ps=1.0, seed=0)

FRAME_ROW = {
    "qm9_index": ("String", None, "the molecule"),
    "generator": ("String", None, "basin / displaced / merged / saddle"),
    "basin": ("Integer", None, "basin index"),
    "k": ("Integer", None, "frame index within the generator"),
    "split": ("String", None, "the Dataset split the frame came from"),
    "distribution": ("String", None, "interpolation / out_of_molecule / in_distribution"),
    "classes": ("String", None, "structure classes, ';'-joined"),
    "n_low": ("Integer", None, "reference modes below the low cutoff"),
    "freq_mae_low_cm": ("Double", "cm^-1", "engine: MAE over the low reference modes"),
    "freq_mae_cm": ("Double", "cm^-1", "engine: MAE over all modes"),
    "hessian_mae": ("Double", "eV/A^2", "engine: element-wise MAE of the Cartesian Hessian"),
    "eigval_mae_eckart": ("Double", "eV/A^2/amu", "engine: MAE of the projected eigenvalues"),
    "mixing": ("Double", None, "engine: off-diagonal weight of D in the reference-mode basis"),
    "loss_cartesian": ("Double", "eV^2/A^4", "the training target exactly: ||H_theta - H_r||_F^2 / (9 N^2) (S0-C-53)"),
    "base_freq_mae_low_cm": ("Double", "cm^-1", "the base model on the same frame"),
    "base_freq_mae_cm": ("Double", "cm^-1", "the base model on the same frame"),
    "base_hessian_mae": ("Double", "eV/A^2", "the base model on the same frame"),
    "base_eigval_mae_eckart": ("Double", "eV/A^2/amu", "the base model on the same frame"),
    "base_loss_cartesian": ("Double", "eV^2/A^4", "the base model on the same frame"),
    "n_anharmonic": ("Integer", None, "reference modes set aside from the entropy tier"),
    "noise_floor_cm": ("Double", "cm^-1", "REF_NOISE_FLOOR_CM: the rigid block of the unprojected REFERENCE Hessian -- a low-mode difference below it is unresolved, not model error (S0-C-44)"),
    "has_hessian": ("Boolean", None, "the frame carries a Label Hessian (the H metrics above are - otherwise)"),
    "rms_displacement_A": ("Double", "A", "the frame's RMS displacement from its basin (0 for a basin frame)"),
    "rms_bin": ("String", None, "0 / <0.08 / <0.15 / >=0.15 (A): the reference row the frame belongs to"),
    "held_out_generator": ("String", None, "yes when the frame's generator never trains (S0-C-54)"),
    "e_err_mev_per_atom": ("Double", "meV/atom", "engine: |E - E_ref| / N"),
    "f_rmse_mev_a": ("Double", "meV/A", "engine: RMSE of the force components against the Label"),
    "base_e_err_mev_per_atom": ("Double", "meV/atom", "the base model on the same frame"),
    "base_f_rmse_mev_a": ("Double", "meV/A", "the base model on the same frame"),
}

SCHEMA = {
    "Calculation_Info": {
        "PROGNAME": ("String", None, "the step that wrote this file"),
        "VERSION": ("String", None, "openQHA version"),
        "STATUS": ("String", None, "the completion marker"),
        "RUN": ("String", None, "the judged run (a train run name, or the engine name)"),
        "TAG": ("String", None, "the campaign tag"),
        "NAME": ("String", None, "the Dataset name"),
        "LEVEL": ("String", None, "the reference level of the Labels"),
        "DATASET_DIR": ("String", None, "the Dataset judged"),
        "ENGINE": ("String", None, "the potential under test"),
        "ENGINE_SCALE": ("Double", None, "1.0, or the calibration scale of a deliberately wrong potential"),
        "BASE_ENGINE": ("String", None, "the potential it is compared against"),
        "MACE_VERSION": ("String", None, "mace.__version__"),
        "MACE_FORK_COMMIT": ("String", None, "commit of the mace checkout, or unknown"),
        "HL_PACKAGE_VERSION": ("String", None, "the openqha-hessian distribution's version (unknown when unreadable)"),
        "HL_PACKAGE_COMMIT": ("String", None, "commit of the package checkout, best-effort (unknown when unreadable)"),
        "SPLITS": ("ArrayOfStrings", None, "the Dataset splits judged"),
        "N_FRAMES": ("Integer", None, "frames with a reference Hessian in those splits"),
        "N_MOLECULES": ("Integer", None, "molecules those frames came from"),
        "LOW_CUTOFF": ("Double", "cm^-1", "a reference mode below this is a low mode"),
        "ANHARMONIC_CM": ("Double", "cm^-1", "below this a mode leaves the entropy tier (round-2 Q6)"),
        "FD_SELF_CHECK_CM": ("Double", "cm^-1", "an along-mode self-check above this leaves it too"),
        "N_HESSIAN_FRAMES": ("Integer", None, "of the frames, those with a Label Hessian"),
        "N_HELD_OUT_FRAMES": ("Integer", None, "of the frames, those of a held-out generator (the reference rows)"),
        "RAMP_MAX_K": ("Double", "K", "the MD ramp's ceiling (0 = the ramp was not run)"),
        "RAMP_STEP_K": ("Double", "K", "the ramp's temperature step"),
        "RAMP_STEP_PS": ("Double", "ps", "the hold at each temperature"),
        "RAMP_SEED": ("Integer", None, "the ramp's velocity / thermostat seed"),
        "N_RAMP_MOLECULES": ("Integer", None, "molecules ramped"),
        "TRAIN_RECORD": ("String", None, "the fine-tune's train.toml whose validation curves head the report, or -"),
        "SECONDS": ("Double", "s", "wall time"),
        "GATE_OPEN": ("Boolean", None, "false (S0-C-60, default): every row reported, nothing decided; true: the gate rows decide"),
        "VERDICT": ("String", None, "REPORTED when the gate is closed; with it open, PASS when every GATE row passed (the Hessian matrix against the Label vs base, in_distribution, forgetting), FAIL otherwise; the reference rows (low modes, entropy, RMS bins, MD ramp) never move it"),
    },
    "Distribution": {
        "DISTRIBUTION": ("String", None, "interpolation / out_of_molecule / in_distribution"),
        "N_FRAMES": ("Integer", None, "frames"),
        "N_MOLECULES": ("Integer", None, "molecules"),
        "FREQ_MAE_LOW_CM": ("Double", "cm^-1", "mean over frames of the low-mode MAE"),
        "FREQ_MAE_CM": ("Double", "cm^-1", "mean over frames of the full-spectrum MAE"),
        "HESSIAN_MAE": ("Double", "eV/A^2", "mean over frames"),
        "EIGVAL_MAE_ECKART": ("Double", "eV/A^2/amu", "mean over frames"),
        "LOSS_CARTESIAN": ("Double", "eV^2/A^4", "mean over frames of the training target ||dH||^2/(9N^2): the gate's number"),
        "BASE_FREQ_MAE_LOW_CM": ("Double", "cm^-1", "the base model, same frames"),
        "BASE_FREQ_MAE_CM": ("Double", "cm^-1", "the base model, same frames"),
        "BASE_HESSIAN_MAE": ("Double", "eV/A^2", "the base model, same frames"),
        "BASE_EIGVAL_MAE_ECKART": ("Double", "eV/A^2/amu", "the base model, same frames"),
        "BASE_LOSS_CARTESIAN": ("Double", "eV^2/A^4", "the base model, same frames"),
    },
    "Class": {
        "CLASS": ("String", None, "the structure class"),
        "N_FRAMES": ("Integer", None, "frames of molecules in this class"),
        "N_MOLECULES": ("Integer", None, "molecules"),
        "FREQ_MAE_LOW_CM": ("Double", "cm^-1", "mean over frames"),
        "FREQ_MAE_CM": ("Double", "cm^-1", "mean over frames"),
        "BASE_FREQ_MAE_LOW_CM": ("Double", "cm^-1", "the base model, same frames"),
        "BASE_FREQ_MAE_CM": ("Double", "cm^-1", "the base model, same frames"),
    },
    "Thermochemistry": {
        "QM9_INDEX": ("String", None, "the molecule"),
        "SOURCE": ("String", None, "the msRRHO Record the numbers were read from, or - "),
        "S_MSRRHO": ("Double", "cal/mol/K", "the engine's msRRHO entropy at its own minima"),
        "MODEL_ERROR_S_REF": ("Double", "cal/mol/K", "engine S_REF - reference S_REF"),
        "N_ANHARMONIC": ("Integer", None, "modes set aside from the entropy tier"),
    },
    "Anharmonic": {
        "QM9_INDEX": ("String", None, "the molecule"),
        "BASIN": ("Integer", None, "basin index"),
        "MODE": ("Integer", None, "index of the reference mode, ascending"),
        "OMEGA_REF_CM": ("Double", "cm^-1", "the reference frequency of that mode"),
        "REASON": ("String", None, "below ANHARMONIC_CM, or an FD self-check above FD_SELF_CHECK_CM"),
    },
    "Forgetting": {
        "FILE": ("String", None, "the fixed SPICE draw"),
        "N_FRAMES": ("Integer", None, "frames evaluated"),
        "ENGINE_E_RMSE_MEV_PER_ATOM": ("Double", "meV/atom", "the potential under test"),
        "ENGINE_F_RMSE_MEV_A": ("Double", "meV/A", "the potential under test"),
        "BASE_E_RMSE_MEV_PER_ATOM": ("Double", "meV/atom", "the base model"),
        "BASE_F_RMSE_MEV_A": ("Double", "meV/A", "the base model"),
        "E_RATIO": ("Double", None, "engine / base"),
        "F_RATIO": ("Double", None, "engine / base"),
    },
    "Displacement": {
        "DISTRIBUTION": ("String", None, "interpolation / out_of_molecule / in_distribution"),
        "RMS_BIN": ("String", None, "0 / <0.08 / <0.15 / >=0.15 A"),
        "N_FRAMES": ("Integer", None, "frames in the bin"),
        "N_MOLECULES": ("Integer", None, "molecules"),
        "N_HESSIAN": ("Integer", None, "of those, frames with a Label Hessian (the H columns' sample)"),
        "HESSIAN_MAE": ("Double", "eV/A^2", "mean over the Hessian frames"),
        "FREQ_MAE_CM": ("Double", "cm^-1", "mean over the Hessian frames"),
        "FREQ_MAE_LOW_CM": ("Double", "cm^-1", "mean over the Hessian frames"),
        "E_MAE_MEV_PER_ATOM": ("Double", "meV/atom", "mean over the frames"),
        "F_RMSE_MEV_A": ("Double", "meV/A", "mean over the frames"),
        "BASE_HESSIAN_MAE": ("Double", "eV/A^2", "the base model, same frames"),
        "BASE_FREQ_MAE_CM": ("Double", "cm^-1", "the base model, same frames"),
        "BASE_FREQ_MAE_LOW_CM": ("Double", "cm^-1", "the base model, same frames"),
        "BASE_E_MAE_MEV_PER_ATOM": ("Double", "meV/atom", "the base model, same frames"),
        "BASE_F_RMSE_MEV_A": ("Double", "meV/A", "the base model, same frames"),
        "GATE": ("String", None, "no: a reference row, never in the verdict"),
    },
    "Ramp": {
        "QM9_INDEX": ("String", None, "the molecule"),
        "WHICH": ("String", None, "engine / base"),
        "SURVIVED": ("Boolean", None, "reached the ceiling without a failure"),
        "FAIL_T_K": ("Double", "K", "the temperature the failure criterion first met (the ceiling when survived)"),
        "FAIL_PS": ("Double", "ps", "simulated time at the failure (the total when survived)"),
        "MAX_RATIO": ("Double", None, "the largest 50-step mean pair distance / equilibrium seen"),
        "MIN_RATIO": ("Double", None, "the smallest"),
        "N_STEPS": ("Integer", None, "MD steps run"),
        "SECONDS": ("Double", "s", "wall time"),
    },
    "Verdict": {
        "LINE": ("String", None, "the row"),
        "GATE": ("String", None, "yes: decides the verdict; no: a reference row, reported only"),
        "VALUE": ("Double", None, "what was measured"),
        "THRESHOLD": ("Double", None, "what it had to beat (a reference row: the base's number)"),
        "RESULT": ("String", None, "PASS / FAIL / -"),
        "NOTE": ("String", None, "how to read it"),
    },
}


class ScaledCalculator:
    """A deliberately wrong potential: forces x `scale`, Hessian x `scale^2`, so every
    frequency is `scale` x the base model's. The judge's must-FAIL calibration -- a
    threshold that this passes is not a threshold."""

    def __init__(self, calc, scale):
        self.calc = calc
        self.scale = float(scale)
        for name in ("r_max", "device", "models"):
            if hasattr(calc, name):
                setattr(self, name, getattr(calc, name))

    def get_hessian(self, atoms=None):
        return np.asarray(self.calc.get_hessian(atoms)) * self.scale ** 2

    def get_forces(self, atoms=None):
        return np.asarray(self.calc.get_forces(atoms)) * self.scale

    def get_potential_energy(self, atoms=None, **kw):
        return float(self.calc.get_potential_energy(atoms, **kw))

    def calculate(self, *a, **k):
        return self.calc.calculate(*a, **k)

    def __getattr__(self, name):
        return getattr(self.calc, name)


def hessian_at(calc, atoms):
    """The engine's full Cartesian Hessian at this geometry, (3N, 3N) eV/A^2 -- the
    shipped path, symmetrised as `hessian_compare` expects."""
    n3 = 3 * len(atoms)
    h = np.asarray(calc.get_hessian(atoms)).reshape(n3, n3)
    return 0.5 * (h + h.T)


def anharmonic_modes(omega_ref_cm, qid, basin, profiles=None):
    """Reference modes the entropy tier must not be judged on (round-2 Q6 (a)): below
    `ANHARMONIC_CM`, or with an along-mode finite-difference self-check above
    `FD_SELF_CHECK_CM` in `profiles` (a `mode_curvature` Record's rows, when one exists)."""
    out = []
    for i, w in enumerate(omega_ref_cm):
        if float(w) < ANHARMONIC_CM:
            out.append(dict(QM9_INDEX=qid, BASIN=int(basin), MODE=int(i), OMEGA_REF_CM=float(w),
                            REASON="omega_r < {:.0f} cm^-1".format(ANHARMONIC_CM)))
            continue
        fd = (profiles or {}).get(i)
        if fd is not None and abs(float(fd)) > FD_SELF_CHECK_CM:
            out.append(dict(QM9_INDEX=qid, BASIN=int(basin), MODE=int(i), OMEGA_REF_CM=float(w),
                            REASON="FD self-check {:.1f} > {:.0f} cm^-1".format(float(fd), FD_SELF_CHECK_CM)))
    return out


def distribution_of(qid, pinned=dataset_mod.PINNED, in_distribution=IN_DISTRIBUTION, molecule_split=None):
    """Which row of the judge's table this molecule belongs to. `in_distribution` wins
    over `out_of_molecule`: a pinned molecule MACE-OFF23 was trained on is not held out,
    whatever the split says (S0-C-40).

    `molecule_split` is the index's column of the same name -- "test" when the WHOLE
    molecule is held out. Under the by-molecule split (production since S0-C-65) every
    test frame belongs to such a molecule, so `interpolation` is empty by construction and
    the held-out rows are a generalisation reading; under the by-frame split a test frame
    of a training molecule is `interpolation` (the same molecule, other conformers), which
    is what round 5 Q4 traded away and S0-C-65 traded back."""
    if qid in in_distribution:
        return "in_distribution"
    if qid in pinned or str(molecule_split or "") == "test":
        return "out_of_molecule"
    return "interpolation"


def rms_bin(rms_A):
    """The reference row a frame belongs to by its RMS displacement from the basin."""
    r = float(rms_A or 0.0)
    if r <= 0.0:
        return RMS_BINS[0][1]
    for edge, label in RMS_BINS[1:]:
        if r < edge:
            return label
    return RMS_BINS[-1][1]


def _ef_errors(c, atoms, e_ref, f_ref):
    """(|E - E_ref| / N in meV/atom, force RMSE in meV/A) of calculator `c` at this frame;
    None where the Label has no energy / forces."""
    at = atoms.copy()
    at.calc = c
    e_err = f_rmse = None
    if e_ref is not None:
        e_err = abs(float(at.get_potential_energy()) - float(e_ref)) / len(at) * 1000.0
    if f_ref is not None:
        d = np.asarray(at.get_forces(), dtype=float) - np.asarray(f_ref, dtype=float)
        f_rmse = float(np.sqrt(np.mean(d * d))) * 1000.0
    return e_err, f_rmse


def frame_rows(dataset_dir, name, level, calc, base_calc=None, splits=("test",), index=None,
               progress=None, energy_key="REF_energy", forces_key="REF_forces"):
    """One row per labelled frame of `splits`: E and F errors of the engine and (when
    given) the base model on every frame, and `hessian_compare` plus eq. 1 exactly where
    the frame carries a Label Hessian (a held-out displaced frame carries E/F only and
    still has a row: the reference bins of ticket 22 read it)."""
    from ase.io import read
    dataset_dir = Path(dataset_dir)
    classes, held, msplit = {}, {}, {}
    if index is None:
        p = dataset_dir / "index.dat"
        index = dat.read_table(p) if p.is_file() else []
    for r in index:
        classes.setdefault(str(r.get("qm9_index")), str(r.get("classes") or "-"))
        msplit.setdefault(str(r.get("qm9_index")), str(r.get("molecule_split") or ""))
        held[(str(r.get("qm9_index")), str(r.get("generator")), int(r.get("basin", 0) or 0), int(r.get("k", 0) or 0))] = \
            str(r.get("held_out_generator", "no"))

    rows, anharmonic = [], []
    for split in splits:
        path = dataset_dir / "{}.{}.extxyz".format(split, level)
        if not path.is_file():
            continue
        for atoms in read(str(path), index=":", format="extxyz"):
            e_ref = atoms.info.get(energy_key)
            if e_ref is None:
                try:
                    e_ref = float(atoms.get_potential_energy())
                except Exception:                                # noqa: BLE001
                    e_ref = None
            f_ref = atoms.arrays.get(forces_key)
            if f_ref is None:
                try:
                    f_ref = atoms.get_forces()
                except Exception:                                # noqa: BLE001
                    f_ref = None
            flat = atoms.info.get("hessian", atoms.info.get("REF_hessian"))
            has_h = flat is not None
            if not has_h and e_ref is None and f_ref is None:
                continue                                         # nothing labelled: nothing to judge
            masses, pos = atoms.get_masses(), atoms.positions
            qid = str(atoms.info.get("qm9_index", "-"))
            gen = str(atoms.info.get("generator", "-"))
            basin, k = int(atoms.info.get("basin", 0)), int(atoms.info.get("k", 0))
            if progress:
                progress("{} {} b{} k{}".format(qid, gen, basin, k))
            rms = float(atoms.info.get("rms_displacement_A", 0.0) or 0.0) if gen != "basin" else 0.0
            row = dict(qm9_index=qid, generator=gen, basin=basin, k=k, split=split,
                       distribution=distribution_of(qid, molecule_split=msplit.get(qid)),
                       classes=classes.get(qid, "-"),
                       has_hessian=bool(has_h), rms_displacement_A=rms, rms_bin=rms_bin(rms),
                       held_out_generator=held.get((qid, gen, basin, k), "yes" if gen != "basin" else "no"),
                       n_low=None, freq_mae_low_cm=None, freq_mae_cm=None, hessian_mae=None,
                       eigval_mae_eckart=None, mixing=None, noise_floor_cm=None,
                       n_anharmonic=0)
            row["e_err_mev_per_atom"], row["f_rmse_mev_a"] = _ef_errors(calc, atoms, e_ref, f_ref)
            if base_calc is not None:
                row["base_e_err_mev_per_atom"], row["base_f_rmse_mev_a"] = _ef_errors(base_calc, atoms, e_ref, f_ref)
            if has_h:
                n3 = 3 * len(atoms)
                h_r = np.asarray(flat, dtype=float).reshape(n3, n3)
                h_e = hessian_at(calc, atoms)
                cmp_e = hc.compare_hessians(h_e, h_r, masses, pos)
                row.update(n_low=int(cmp_e["N_LOW"]),
                           freq_mae_low_cm=float(cmp_e["FREQ_MAE_LOW_CM"]),
                           freq_mae_cm=float(cmp_e["FREQ_MAE_CM"]),
                           hessian_mae=float(cmp_e["HESSIAN_MAE"]),
                           eigval_mae_eckart=float(cmp_e["EIGVAL_MAE_ECKART"]),
                           mixing=float(cmp_e["MIXING"]),
                           loss_cartesian=phl.cartesian_loss_full(h_e, h_r),
                           noise_floor_cm=float(cmp_e.get("REF_NOISE_FLOOR_CM", 0.0)))
                anh = anharmonic_modes(cmp_e["OMEGA_REF_CM"], qid, basin)
                row["n_anharmonic"] = len(anh)
                if gen == "basin":
                    anharmonic.extend(anh)
                if base_calc is not None:
                    h_b = hessian_at(base_calc, atoms)
                    cmp_b = hc.compare_hessians(h_b, h_r, masses, pos)
                    row.update(base_freq_mae_low_cm=float(cmp_b["FREQ_MAE_LOW_CM"]),
                               base_freq_mae_cm=float(cmp_b["FREQ_MAE_CM"]),
                               base_hessian_mae=float(cmp_b["HESSIAN_MAE"]),
                               base_eigval_mae_eckart=float(cmp_b["EIGVAL_MAE_ECKART"]),
                               base_loss_cartesian=phl.cartesian_loss_full(h_b, h_r))
            rows.append(row)
    return rows, anharmonic


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def aggregate(rows):
    """The two tables the judge is read from: per distribution and per class. Both are
    the HESSIAN frames' tables (a frame without a Label Hessian has no frequency to
    average); the E/F-only frames are read by `aggregate_displacement`."""
    rows = [r for r in rows if r.get("has_hessian", True)]
    dist_rows = []
    for d in ("interpolation", "out_of_molecule", "in_distribution"):
        sel = [r for r in rows if r["distribution"] == d]
        if not sel:
            continue
        dist_rows.append(dict(
            DISTRIBUTION=d, N_FRAMES=len(sel), N_MOLECULES=len({r["qm9_index"] for r in sel}),
            FREQ_MAE_LOW_CM=_mean(sel, "freq_mae_low_cm"), FREQ_MAE_CM=_mean(sel, "freq_mae_cm"),
            HESSIAN_MAE=_mean(sel, "hessian_mae"), EIGVAL_MAE_ECKART=_mean(sel, "eigval_mae_eckart"),
            LOSS_CARTESIAN=_mean(sel, "loss_cartesian"),
            BASE_FREQ_MAE_LOW_CM=_mean(sel, "base_freq_mae_low_cm"),
            BASE_FREQ_MAE_CM=_mean(sel, "base_freq_mae_cm"),
            BASE_HESSIAN_MAE=_mean(sel, "base_hessian_mae"),
            BASE_EIGVAL_MAE_ECKART=_mean(sel, "base_eigval_mae_eckart"),
            BASE_LOSS_CARTESIAN=_mean(sel, "base_loss_cartesian")))
    names = sorted({c for r in rows for c in str(r.get("classes", "-")).split(";") if c and c != "-"})
    cls_rows = []
    for c in names:
        sel = [r for r in rows if c in str(r.get("classes", "-")).split(";")]
        cls_rows.append(dict(CLASS=c, N_FRAMES=len(sel), N_MOLECULES=len({r["qm9_index"] for r in sel}),
                             FREQ_MAE_LOW_CM=_mean(sel, "freq_mae_low_cm"), FREQ_MAE_CM=_mean(sel, "freq_mae_cm"),
                             BASE_FREQ_MAE_LOW_CM=_mean(sel, "base_freq_mae_low_cm"),
                             BASE_FREQ_MAE_CM=_mean(sel, "base_freq_mae_cm")))
    return dist_rows, cls_rows


def aggregate_displacement(rows):
    """The reference rows of ticket 22: every frame (Hessian or E/F-only) grouped by
    (distribution, rms_bin) in bin order; H metrics over the Hessian frames of the bin, E
    and F over all of them; `GATE = no` on every row."""
    out = []
    for d in ("interpolation", "out_of_molecule", "in_distribution"):
        for _edge, label in RMS_BINS:
            sel = [r for r in rows if r["distribution"] == d and r.get("rms_bin", rms_bin(r.get("rms_displacement_A", 0.0))) == label]
            if not sel:
                continue
            hs = [r for r in sel if r.get("has_hessian", r.get("hessian_mae") is not None)]
            out.append(dict(
                DISTRIBUTION=d, RMS_BIN=label, N_FRAMES=len(sel), N_MOLECULES=len({r["qm9_index"] for r in sel}),
                N_HESSIAN=len(hs),
                HESSIAN_MAE=_mean(hs, "hessian_mae"), FREQ_MAE_CM=_mean(hs, "freq_mae_cm"),
                FREQ_MAE_LOW_CM=_mean(hs, "freq_mae_low_cm"),
                E_MAE_MEV_PER_ATOM=_mean(sel, "e_err_mev_per_atom"), F_RMSE_MEV_A=_mean(sel, "f_rmse_mev_a"),
                BASE_HESSIAN_MAE=_mean(hs, "base_hessian_mae"), BASE_FREQ_MAE_CM=_mean(hs, "base_freq_mae_cm"),
                BASE_FREQ_MAE_LOW_CM=_mean(hs, "base_freq_mae_low_cm"),
                BASE_E_MAE_MEV_PER_ATOM=_mean(sel, "base_e_err_mev_per_atom"),
                BASE_F_RMSE_MEV_A=_mean(sel, "base_f_rmse_mev_a"), GATE="no"))
    return out


def ramp_schedule(start_K=RAMP["start_K"], step_K=RAMP["step_K"], max_K=RAMP["max_K"]):
    """The temperatures of the ramp: start, start + step, ..., up to and including max_K."""
    out, t = [], float(start_K)
    while t <= float(max_K) + 1e-9:
        out.append(t)
        t += float(step_K)
    return out


def ramp_one(calc, atoms, max_K=RAMP["max_K"], start_K=RAMP["start_K"], step_K=RAMP["step_K"],
             step_ps=RAMP["step_ps"], timestep_fs=RAMP["timestep_fs"], window_steps=RAMP["window_steps"],
             upper=RAMP["upper"], lower=RAMP["lower"], seed=RAMP["seed"], optimise=True, fmax=0.005):
    """Rodriguez 2025's MD temperature ramp on one molecule with one calculator: from
    the calculator's OWN minimum (BFGS from `atoms`, unless `optimise` is False), Langevin
    (`s0_B_md_stability.run_one`'s machinery: 1 fs, friction 1/ps, seeded velocities),
    `start_K` then +`step_K` every `step_ps`, until any atom pair's `window_steps`-step
    mean distance leaves [`lower`, `upper`] x its equilibrium value or the ceiling is
    reached. Returns the `[Ramp]` row fields without QM9_INDEX / WHICH."""
    import time
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.optimize import BFGS
    from scipy.spatial.distance import pdist
    t0 = time.time()
    atoms = atoms.copy()
    atoms.calc = calc
    if optimise:
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=300)
    d_eq = pdist(atoms.get_positions())
    rng = np.random.RandomState(int(seed))
    MaxwellBoltzmannDistribution(atoms, temperature_K=float(start_K), rng=rng)
    dyn = Langevin(atoms, float(timestep_fs) * units.fs, temperature_K=float(start_K),
                   friction=RAMP["friction_per_ps"] / (1000.0 * units.fs), rng=rng, fixcm=False)
    steps_per_stage = max(1, int(round(float(step_ps) * 1000.0 / float(timestep_fs))))
    window = max(1, min(int(window_steps), steps_per_stage))
    state = dict(sum=np.zeros_like(d_eq), n=0, max_ratio=1.0, min_ratio=1.0, fail_T=None, fail_step=None, T=float(start_K))

    def watch():
        state["sum"] += pdist(atoms.get_positions())
        state["n"] += 1
        if state["n"] >= window:
            ratio = state["sum"] / state["n"] / d_eq
            state["max_ratio"] = max(state["max_ratio"], float(ratio.max()))
            state["min_ratio"] = min(state["min_ratio"], float(ratio.min()))
            if state["fail_T"] is None and (ratio.max() > upper or ratio.min() < lower):
                state["fail_T"], state["fail_step"] = state["T"], dyn.get_number_of_steps()
            state["sum"][:] = 0.0
            state["n"] = 0

    dyn.attach(watch, interval=1)
    n_steps = 0
    for T in ramp_schedule(start_K, step_K, max_K):
        state["T"] = T
        dyn.set_temperature(temperature_K=T)
        done = 0
        chunk = window
        while done < steps_per_stage and state["fail_T"] is None:
            step = min(chunk, steps_per_stage - done)
            dyn.run(step)
            done += step
        n_steps += done
        if state["fail_T"] is not None:
            break
    survived = state["fail_T"] is None
    return dict(SURVIVED=bool(survived),
                FAIL_T_K=float(max_K) if survived else float(state["fail_T"]),
                FAIL_PS=float(n_steps * timestep_fs / 1000.0) if survived else float(state["fail_step"] * timestep_fs / 1000.0),
                MAX_RATIO=float(state["max_ratio"]), MIN_RATIO=float(state["min_ratio"]),
                N_STEPS=int(n_steps), SECONDS=float(time.time() - t0))


def md_ramp(calc, base_calc, molecules, geometries, progress=None, **ramp):
    """The `[Ramp]` rows: `ramp_one` per molecule for the engine and for the base (when
    given), from `geometries[qid]` (an Atoms: the molecule's basin frame). `ramp` keys as
    `ramp_one`'s (max_K, step_K, step_ps, ...)."""
    rows = []
    for qid in molecules:
        at = geometries.get(qid)
        if at is None:
            continue
        for which, c in (("engine", calc), ("base", base_calc)):
            if c is None:
                continue
            if progress:
                progress("ramp {} {}".format(qid, which))
            r = ramp_one(c, at, **ramp)
            rows.append(dict(QM9_INDEX=qid, WHICH=which, **r))
    return rows


def basin_geometries(dataset_dir, level, molecules, splits=("test", "train", "valid")):
    """{qid: Atoms} -- each molecule's basin-0 frame from the Dataset's split files (the
    first basin frame seen), the starting point of its ramp."""
    from ase.io import read
    out = {}
    want = set(molecules)
    for split in splits:
        path = Path(dataset_dir) / "{}.{}.extxyz".format(split, level)
        if not path.is_file():
            continue
        for atoms in read(str(path), index=":", format="extxyz"):
            qid = str(atoms.info.get("qm9_index", "-"))
            if qid in want and qid not in out and str(atoms.info.get("generator")) == "basin":
                a = atoms.copy()
                a.calc = None
                out[qid] = a
        if want <= set(out):
            break
    return out


def training_curves(train_toml):
    """The three validation curves of a fine-tune's Record (ticket 21), for the report's
    header: rows (epoch, valid_energy, valid_forces, valid_hessian) of the valid split,
    and the Record's HESSIAN_CURVE_MOVED; ({}, []) when no Record."""
    path = Path(train_toml) if train_toml else None
    if not path or not path.is_file():
        return {}, []
    rec = prop.load(path)
    info = rec.get("Calculation_Info", {})
    rows = [dict(epoch=int(r["EPOCH"]), valid_energy=r.get("VALID_ENERGY"), valid_forces=r.get("VALID_FORCES"),
                 valid_hessian=r.get("VALID_HESSIAN"))
            for r in rec.get("Epoch", []) if r.get("SPLIT") == "valid"]
    return info, rows


def thermochemistry(root, tag, molecules, engine_level, reference_level, anharmonic_rows=()):
    """The entropy tier, READ from the msRRHO Records on disk -- never recomputed here.
    A judge that produced the numbers it judges would be marking its own work; running
    the pipeline (`s0_thermo_msrrho.py` with `S0_ENGINE=<engine>`) is a separate step,
    and a molecule whose Record is absent is reported as absent."""
    n_anh = {}
    for a in anharmonic_rows:
        n_anh[a["QM9_INDEX"]] = n_anh.get(a["QM9_INDEX"], 0) + 1
    out = []
    for qid in molecules:
        mol = layout.molecule_dir(root, tag, qid)
        path = layout.level_file(mol, reference_level, "thermo_msrrho.toml")
        row = dict(QM9_INDEX=qid, SOURCE="-", S_MSRRHO=None, MODEL_ERROR_S_REF=None,
                   N_ANHARMONIC=int(n_anh.get(qid, 0)))
        if Path(path).is_file():
            rec = prop.load(path)
            info = rec.get("Calculation_Info", {})
            row["SOURCE"] = str(path)
            for key in ("S_MSRRHO", "S_TOTAL", "S_ABS"):
                if info.get(key) is not None:
                    row["S_MSRRHO"] = float(info[key])
                    break
            if info.get("MODEL_ERROR_S_REF") is not None:
                row["MODEL_ERROR_S_REF"] = float(info["MODEL_ERROR_S_REF"])
        out.append(row)
    return out


def forgetting(calc, base_calc, frames_file, energy_key="REF_energy", forces_key="REF_forces"):
    """E and F RMSE of the engine and of the base model on a FIXED set of SPICE frames
    (round-2 Q7's judge). Energies per atom, so molecules of different size compare."""
    from ase.io import read
    path = Path(frames_file)
    if not path.is_file():
        return None
    frames = read(str(path), index=":", format="extxyz")
    acc = {"engine": ([], []), "base": ([], [])}
    for atoms in frames:
        e_ref = atoms.info.get(energy_key)
        if e_ref is None:
            try:
                e_ref = float(atoms.get_potential_energy())
            except Exception:                                    # noqa: BLE001
                continue
        f_ref = atoms.arrays.get(forces_key)
        if f_ref is None:
            try:
                f_ref = atoms.get_forces()
            except Exception:                                    # noqa: BLE001
                f_ref = None
        for which, c in (("engine", calc), ("base", base_calc)):
            at = atoms.copy()
            at.calc = c
            de = (float(at.get_potential_energy()) - float(e_ref)) / len(at)
            acc[which][0].append(de)
            if f_ref is not None:
                acc[which][1].append(np.asarray(at.get_forces()) - np.asarray(f_ref))
    def rmse(v):
        return float(np.sqrt(np.mean(np.concatenate([np.asarray(x).reshape(-1) for x in v]) ** 2))) if v else None
    e_eng, e_base = rmse(acc["engine"][0]), rmse(acc["base"][0])
    f_eng, f_base = rmse(acc["engine"][1]), rmse(acc["base"][1])
    return dict(FILE=str(path), N_FRAMES=len(frames),
                ENGINE_E_RMSE_MEV_PER_ATOM=None if e_eng is None else e_eng * 1000.0,
                ENGINE_F_RMSE_MEV_A=None if f_eng is None else f_eng * 1000.0,
                BASE_E_RMSE_MEV_PER_ATOM=None if e_base is None else e_base * 1000.0,
                BASE_F_RMSE_MEV_A=None if f_base is None else f_base * 1000.0,
                E_RATIO=None if not e_base else e_eng / e_base,
                F_RATIO=None if not f_base else f_eng / f_base)


def verdict(dist_rows, thermo_rows, forget_row, thresholds=None, noise_floor_cm=None, ramp_rows=None,
            disp_rows=None):
    """One line per row. GATE rows (`GATE_ROWS`, S0-C-58/59) decide: the Hessian MATRIX
    against the Label on the held-out Hessian frames -- ||H_theta - H_r||_F^2 / (9 N^2),
    the training target's own number -- engine against base (no worse); the
    in_distribution no-degradation; the forgetting line. REFERENCE rows
    (`REFERENCE_ROWS`) are what is computed from the matrix afterwards and the
    extrapolation readouts: the low-mode frequency line, the msRRHO entropy at the engine's
    own minima, the held-out generator's Hessian error against the base (the RMS bins),
    the MD ramp -- measured against their number, reported, never moving the verdict. A
    line with nothing to measure is `-`, never a silent PASS. The Label's grid noise
    (S0-C-44) is printed beside the low-mode line: no threshold means anything below it."""
    t = dict(THRESHOLDS, **(thresholds or {}))
    lines = []
    held = [d for d in dist_rows if d["DISTRIBUTION"] in ("interpolation", "out_of_molecule")]
    # --- gate rows ----------------------------------------------------------------------
    cart = [(d["LOSS_CARTESIAN"], d.get("BASE_LOSS_CARTESIAN"), d["N_FRAMES"]) for d in held
            if d.get("LOSS_CARTESIAN") is not None]
    if cart:
        n_all = sum(n for _e, _b, n in cart)
        eng = sum(e * n for e, _b, n in cart) / n_all
        if all(b is not None for _e, b, _n in cart):
            base = sum(b * n for _e, b, n in cart) / n_all
            v = eng / base - 1.0 if base else None
            lines.append(dict(LINE="held_out_hessian_cartesian", GATE="yes", VALUE=v, THRESHOLD=t["hessian_cartesian_vs_base"],
                              RESULT=("-" if v is None else ("PASS" if v <= t["hessian_cartesian_vs_base"] else "FAIL")),
                              NOTE="the training target on the held-out Hessian frames: engine ||dH||^2/(9N^2) = {:.4e} against the "
                                   "base's {:.4e} (ratio - 1; a negative number is an improvement)".format(eng, base)))
        else:
            lines.append(dict(LINE="held_out_hessian_cartesian", GATE="yes", VALUE=None, THRESHOLD=t["hessian_cartesian_vs_base"],
                              RESULT="-", NOTE="engine ||dH||^2/(9N^2) = {:.4e}; no base model to compare against".format(eng)))
    ind = [d for d in dist_rows if d["DISTRIBUTION"] == "in_distribution"]
    if ind and ind[0].get("BASE_FREQ_MAE_CM") is not None:
        d = ind[0]
        worst, worst_name = None, "-"
        for key in ("FREQ_MAE_LOW_CM", "FREQ_MAE_CM", "HESSIAN_MAE", "EIGVAL_MAE_ECKART"):
            base = d.get("BASE_" + key)
            if base:
                ratio = d[key] / base - 1.0
                if worst is None or ratio > worst:
                    worst, worst_name = ratio, key
        lines.append(dict(LINE="in_distribution_degradation", GATE="yes", VALUE=worst,
                          THRESHOLD=t["in_distribution_degradation"],
                          RESULT="PASS" if worst is not None and worst <= t["in_distribution_degradation"] else "FAIL",
                          NOTE="worst HIP metric against the base model ({}); a negative number is an improvement".format(worst_name)))
    if forget_row and forget_row.get("F_RATIO") is not None:
        v = max(forget_row["F_RATIO"], forget_row.get("E_RATIO") or 0.0) - 1.0
        lines.append(dict(LINE="forgetting", GATE="yes", VALUE=v, THRESHOLD=t["forgetting"],
                          RESULT="PASS" if v <= t["forgetting"] else "FAIL",
                          NOTE="SPICE E/F RMSE against the base model's on {} frames (<= 1.15x)".format(forget_row["N_FRAMES"])))
    else:
        lines.append(dict(LINE="forgetting", GATE="yes", VALUE=None, THRESHOLD=t["forgetting"], RESULT="-",
                          NOTE="no SPICE draw on disk (scripts/tooling/s0_spice_test_draw.py)"))
    # --- reference rows (computed from the matrix afterwards; post-processing; S0-C-58/59) ---
    vals = [d["FREQ_MAE_LOW_CM"] for d in held if d.get("FREQ_MAE_LOW_CM") is not None]
    if vals:
        v = float(np.mean(vals))
        note = "reference: held-out low-mode MAE (< {:.0f} cm^-1 modes), computed from the matrix".format(LOW_CM)
        if noise_floor_cm:
            note += "; the Label's own grid noise is ~{:.0f} cm^-1 (S0-C-44)".format(noise_floor_cm)
        lines.append(dict(LINE="held_out_low_mode_mae_cm", GATE="no", VALUE=v, THRESHOLD=t["low_mode_mae_cm"],
                          RESULT="PASS" if v <= t["low_mode_mae_cm"] else "FAIL", NOTE=note))
    errs = [abs(r["MODEL_ERROR_S_REF"]) for r in thermo_rows if r.get("MODEL_ERROR_S_REF") is not None]
    lines.append(dict(LINE="model_error_s_ref_cal_per_mol_K", GATE="no", VALUE=max(errs) if errs else None,
                      THRESHOLD=t["model_error_s_ref"],
                      RESULT=("-" if not errs else ("PASS" if max(errs) <= t["model_error_s_ref"] else "FAIL")),
                      NOTE="reference: largest |engine - reference| msRRHO entropy error at the engine's own minima over {} pinned "
                           "molecule(s), anharmonic modes excluded (computed from the matrix); '-' means no msRRHO Record on disk".format(len(errs))))
    disp = [d for d in (disp_rows or []) if d["RMS_BIN"] != RMS_BINS[0][1] and d.get("HESSIAN_MAE") is not None
            and d.get("BASE_HESSIAN_MAE")]
    if disp:
        eng = float(np.mean([d["HESSIAN_MAE"] for d in disp]))
        base = float(np.mean([d["BASE_HESSIAN_MAE"] for d in disp]))
        lines.append(dict(LINE="held_out_generator_hessian_vs_base", GATE="no", VALUE=eng / base - 1.0 if base else None,
                          THRESHOLD=0.0, RESULT="PASS" if base and eng <= base else "FAIL",
                          NOTE="reference: Hessian MAE on the held-out generator's frames (rms > 0) against the base's "
                               "({} bins); a negative number is an improvement away from the minimum".format(len(disp))))
    if ramp_rows:
        eng = [r["FAIL_T_K"] for r in ramp_rows if r["WHICH"] == "engine"]
        base = [r["FAIL_T_K"] for r in ramp_rows if r["WHICH"] == "base"]
        if eng:
            v = float(min(eng))
            b = float(min(base)) if base else None
            lines.append(dict(LINE="md_ramp_K", GATE="no", VALUE=v, THRESHOLD=b,
                              RESULT=("-" if b is None else ("PASS" if v >= b else "FAIL")),
                              NOTE="reference: the lowest failure temperature of the MD ramp over {} molecule(s), engine "
                                   "against base (the ceiling when no failure)".format(len(eng))))
    return lines


#: the gate is CLOSED by default (S0-C-60): every row is reported against its number,
#: none decides; `judge.run(gate=True)` / `06_judge.py --gate` reopens it
GATE_CLOSED = "REPORTED"


def verdict_of(lines, gate=True):
    """With the gate open: PASS when every GATE row is PASS or '-', FAIL otherwise; a
    reference row never moves it. With the gate closed (the default of `run`, S0-C-60):
    `REPORTED` -- the rows carry their own PASS / FAIL and nothing is decided."""
    if not gate:
        return GATE_CLOSED
    return "PASS" if all(l["RESULT"] != "FAIL" for l in lines if l.get("GATE", "yes") == "yes") else "FAIL"


def run(root, tag, name, level, calc, engine_name, base_calc=None, base_engine=None, run_name=None,
        splits=("test",), scale=1.0, spice_file=None, thresholds=None, progress=None,
        reference_level=None, write=True,
        ramp=None, ramp_molecules=None, train_record=None, gate=False, thermo_tag=None):
    """The whole judge: frames, aggregation, the entropy tier, forgetting, the reference
    bins, the MD ramp (`ramp`: a dict of `ramp_one` settings, None = not run;
    `ramp_molecules`: default the pinned molecules present), the verdict, and the Record
    under `<dataset>/judge/<run>/`. `train_record`: the fine-tune's train.toml whose
    validation curves head the report. `gate`: False (S0-C-60, the default) reports every
    row and decides nothing (`VERDICT = REPORTED`); True lets the gate rows decide.
    `thermo_tag`: the tag whose molecule directories hold the engine's msRRHO Records (a
    fine-tuned engine has its own branch A under its own tag); default the campaign tag."""
    import time
    t0 = time.time()
    dataset_dir = Path(dataset_mod.datasets_dir(root, tag, name))
    run_name = run_name or engine_name
    rows, anharmonic = frame_rows(dataset_dir, name, level, calc, base_calc=base_calc,
                                  splits=splits, progress=progress)
    if not rows:
        # A judge with nothing to judge must not answer PASS. The most common cause is a
        # Dataset that has no labelled frames in these splits (a by-molecule smoke set
        # puts them all in `test`, a fresh campaign in `pool`), or a path that is not a
        # Dataset directory at all.
        raise ValueError(
            "no labelled frame in {} for splits {} at level {}: nothing to judge. "
            "Check `04_dataset.py` put frames there (`index.dat` lists their splits) and that the "
            "level is the one the Labels were made at.".format(dataset_dir, ", ".join(splits), level))
    dist_rows, cls_rows = aggregate(rows)
    disp_rows = aggregate_displacement(rows)
    molecules = sorted({r["qm9_index"] for r in rows})
    thermo_rows = thermochemistry(root, thermo_tag or tag, molecules, engine_name, reference_level or level,
                                  anharmonic_rows=anharmonic)
    forget_row = forgetting(calc, base_calc, spice_file) if (spice_file and base_calc is not None) else None
    ramp_rows = []
    ramp = dict(ramp) if ramp else None
    if ramp is not None:
        wanted = list(ramp_molecules) if ramp_molecules else [q for q in molecules if q in dataset_mod.PINNED]
        geoms = basin_geometries(dataset_dir, level, wanted)
        ramp_rows = md_ramp(calc, base_calc, wanted, geoms, progress=progress, **ramp)
    floors = [r["noise_floor_cm"] for r in rows if r.get("noise_floor_cm")]
    lines = verdict(dist_rows, thermo_rows, forget_row, thresholds,
                    noise_floor_cm=max(floors) if floors else None, ramp_rows=ramp_rows, disp_rows=disp_rows)
    train_info, curves = training_curves(train_record)

    import mace
    from openqha.potentials import engine as engine_mod
    fork = engine_mod.mace_fork_info()
    identity = package_identity()
    info = dict(RUN=str(run_name), TAG=str(tag), NAME=str(name), LEVEL=str(level),
                DATASET_DIR=str(dataset_dir), ENGINE=str(engine_name),
                ENGINE_SCALE=float(scale), BASE_ENGINE=str(base_engine or "-"),
                MACE_VERSION=mace.__version__, MACE_FORK_COMMIT=fork["mace_fork_commit"],
                HL_PACKAGE_VERSION=identity["HL_PACKAGE_VERSION"], HL_PACKAGE_COMMIT=identity["HL_PACKAGE_COMMIT"],
                SPLITS=[str(s) for s in splits], N_FRAMES=len(rows), N_MOLECULES=len(molecules),
                LOW_CUTOFF=float(LOW_CM), ANHARMONIC_CM=float(ANHARMONIC_CM),
                FD_SELF_CHECK_CM=float(FD_SELF_CHECK_CM),
                N_HESSIAN_FRAMES=sum(1 for r in rows if r.get("has_hessian", True)),
                N_HELD_OUT_FRAMES=sum(1 for r in rows if r.get("held_out_generator") == "yes"),
                RAMP_MAX_K=float(ramp.get("max_K", RAMP["max_K"])) if ramp is not None else 0.0,
                RAMP_STEP_K=float(ramp.get("step_K", RAMP["step_K"])) if ramp is not None else 0.0,
                RAMP_STEP_PS=float(ramp.get("step_ps", RAMP["step_ps"])) if ramp is not None else 0.0,
                RAMP_SEED=int(ramp.get("seed", RAMP["seed"])) if ramp is not None else 0,
                N_RAMP_MOLECULES=len({r["QM9_INDEX"] for r in ramp_rows}),
                TRAIN_RECORD=str(train_record) if train_record else "-",
                SECONDS=float(time.time() - t0),
                GATE_OPEN=bool(gate),
                VERDICT=verdict_of(lines, gate=gate))
    out = dict(info=info, frames=rows, distributions=dist_rows, classes=cls_rows, displacement=disp_rows,
               thermochemistry=thermo_rows, anharmonic=anharmonic, forgetting=forget_row, ramp=ramp_rows,
               verdict=lines, train_info=train_info, curves=curves, run_dir=dataset_dir / STEP / run_name)
    if write:
        write_record(out)
    return out


def write_record(out):
    d = Path(out["run_dir"])
    d.mkdir(parents=True, exist_ok=True)
    blocks = {"Calculation_Info": out["info"], "Distribution": out["distributions"],
              "Class": out["classes"], "Displacement": out.get("displacement", []),
              "Thermochemistry": out["thermochemistry"],
              "Anharmonic": out["anharmonic"],
              "Forgetting": [out["forgetting"]] if out["forgetting"] else [],
              "Ramp": out.get("ramp", []),
              "Verdict": out["verdict"]}
    missing = prop.write(d / (STEP + ".toml"), blocks, SCHEMA, prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("judge.toml keys outside the schema: {}".format(missing))
    dat.write_table(d / (STEP + ".dat"), out["frames"], list(FRAME_ROW), FRAME_ROW)
    _write_report(d / (STEP + ".out"), out)


def _num(x, fmt="{:.3f}"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def _write_report(path, out):
    info = out["info"]
    rep = report.Report(PROGNAME, "Judge of {!r} on {} at {}".format(info["ENGINE"], info["NAME"], info["LEVEL"]))
    rep.section("what was judged")
    for k in ("RUN", "TAG", "NAME", "LEVEL", "DATASET_DIR", "SPLITS", "N_FRAMES", "N_HESSIAN_FRAMES",
              "N_HELD_OUT_FRAMES", "N_MOLECULES", "ENGINE", "ENGINE_SCALE", "BASE_ENGINE",
              "MACE_VERSION", "MACE_FORK_COMMIT", "HL_PACKAGE_VERSION", "HL_PACKAGE_COMMIT",
              "LOW_CUTOFF", "ANHARMONIC_CM", "SECONDS"):
        rep.kv(k, info.get(k))
    curves = out.get("curves") or []
    if curves:
        ti = out.get("train_info") or {}
        rep.section("the fine-tune's validation curves (ticket 21's Record: {})".format(info.get("TRAIN_RECORD")))
        for k in ("VALID_PROBES", "HESSIAN_WEIGHT", "PT_N_FRAMES", "REPLAY_PER_HESSIAN_FRAME",
                  "STAGE_TWO_EPOCH", "HESSIAN_CURVE_MOVED", "N_EPOCHS"):
            if k in ti:
                rep.kv(k, ti.get(k))
        shown = curves if len(curves) <= 8 else curves[:2] + curves[-6:]
        rep.table(["epoch", "valid E term", "valid F term", "valid H term"],
                  [[r["epoch"], _num(r["valid_energy"], "{:.4e}"), _num(r["valid_forces"], "{:.4e}"),
                    _num(r["valid_hessian"], "{:.4e}")] for r in shown])
    rep.section("per distribution (the engine, then the base model)")
    rep.table(["distribution", "frames", "mols", "low MAE", "MAE", "||dH||^2/9N^2", "base low", "base MAE", "base ||dH||^2/9N^2"],
              [[d["DISTRIBUTION"], d["N_FRAMES"], d["N_MOLECULES"], _num(d["FREQ_MAE_LOW_CM"], "{:.2f}"),
                _num(d["FREQ_MAE_CM"], "{:.2f}"), _num(d["LOSS_CARTESIAN"], "{:.4e}"),
                _num(d["BASE_FREQ_MAE_LOW_CM"], "{:.2f}"), _num(d["BASE_FREQ_MAE_CM"], "{:.2f}"),
                _num(d["BASE_LOSS_CARTESIAN"], "{:.4e}")] for d in out["distributions"]],
              units=["", "", "", "cm^-1", "cm^-1", "eV^2/A^4", "cm^-1", "cm^-1", "eV^2/A^4"])
    if out["classes"]:
        rep.section("per structure class")
        rep.table(["class", "frames", "mols", "low MAE", "MAE", "base low", "base MAE"],
                  [[c["CLASS"], c["N_FRAMES"], c["N_MOLECULES"], _num(c["FREQ_MAE_LOW_CM"], "{:.2f}"),
                    _num(c["FREQ_MAE_CM"], "{:.2f}"), _num(c["BASE_FREQ_MAE_LOW_CM"], "{:.2f}"),
                    _num(c["BASE_FREQ_MAE_CM"], "{:.2f}")] for c in out["classes"]],
                  units=["", "", "", "cm^-1", "cm^-1", "cm^-1", "cm^-1"])
    if out["thermochemistry"]:
        rep.section("thermochemistry at the engine's own minima (READ from the msRRHO Records, not recomputed)")
        rep.table(["molecule", "S_msRRHO", "model error", "anharmonic modes", "source"],
                  [[r["QM9_INDEX"], _num(r["S_MSRRHO"], "{:.3f}"), _num(r["MODEL_ERROR_S_REF"], "{:+.3f}"),
                    r["N_ANHARMONIC"], r["SOURCE"]] for r in out["thermochemistry"]],
                  units=["", "cal/mol/K", "cal/mol/K", "", ""])
    if out["anharmonic"]:
        rep.section("modes set aside from the entropy tier (round-2 Q6)")
        rep.table(["molecule", "basin", "mode", "omega_ref", "reason"],
                  [[a["QM9_INDEX"], a["BASIN"], a["MODE"], _num(a["OMEGA_REF_CM"], "{:.2f}"), a["REASON"]]
                   for a in out["anharmonic"]], units=["", "", "", "cm^-1", ""])
    if out["forgetting"]:
        rep.section("forgetting (a fixed SPICE draw)")
        for k, v in out["forgetting"].items():
            rep.kv(k, v if not isinstance(v, float) else round(v, 4))
    if out.get("displacement"):
        rep.section("reference rows: the held-out generator's frames by RMS displacement (never gated; S0-C-54)")
        rep.table(["distribution", "rms bin", "frames", "mols", "H frames", "H MAE", "MAE", "E MAE", "F RMSE",
                   "base H MAE", "base MAE", "base E MAE", "base F RMSE"],
                  [[d["DISTRIBUTION"], d["RMS_BIN"], d["N_FRAMES"], d["N_MOLECULES"], d["N_HESSIAN"],
                    _num(d["HESSIAN_MAE"], "{:.4f}"), _num(d["FREQ_MAE_CM"], "{:.2f}"),
                    _num(d["E_MAE_MEV_PER_ATOM"], "{:.2f}"), _num(d["F_RMSE_MEV_A"], "{:.2f}"),
                    _num(d["BASE_HESSIAN_MAE"], "{:.4f}"), _num(d["BASE_FREQ_MAE_CM"], "{:.2f}"),
                    _num(d["BASE_E_MAE_MEV_PER_ATOM"], "{:.2f}"), _num(d["BASE_F_RMSE_MEV_A"], "{:.2f}")]
                   for d in out["displacement"]],
                  units=["", "A", "", "", "", "eV/A^2", "cm^-1", "meV/atom", "meV/A", "eV/A^2", "cm^-1", "meV/atom", "meV/A"])
    if out.get("ramp"):
        rep.section("reference row: the MD temperature ramp (Rodriguez 2025; +{:.0f} K every {:.0f} ps from 5 K to {:.0f} K, seed {})".format(
            info.get("RAMP_STEP_K", 0.0), info.get("RAMP_STEP_PS", 0.0), info.get("RAMP_MAX_K", 0.0), info.get("RAMP_SEED", 0)))
        rep.table(["molecule", "which", "survived", "fail T", "fail time", "max ratio", "min ratio", "steps", "seconds"],
                  [[r["QM9_INDEX"], r["WHICH"], "yes" if r["SURVIVED"] else "no", _num(r["FAIL_T_K"], "{:.0f}"),
                    _num(r["FAIL_PS"], "{:.2f}"), _num(r["MAX_RATIO"], "{:.3f}"), _num(r["MIN_RATIO"], "{:.3f}"),
                    r["N_STEPS"], _num(r["SECONDS"], "{:.0f}")] for r in out["ramp"]],
                  units=["", "", "", "K", "ps", "", "", "", "s"])
    rep.section("verdict (gate {}: {})".format("open -- the gate rows decide" if info.get("GATE_OPEN") else "CLOSED, S0-C-60",
                                             "reference rows are reported" if info.get("GATE_OPEN") else "every row is reported, none decides"))
    rep.table(["line", "gate", "value", "threshold", "result", "note"],
              [[l["LINE"], l.get("GATE", "yes"), _num(l["VALUE"], "{:.4f}"), _num(l["THRESHOLD"], "{:.4f}"), l["RESULT"], l["NOTE"]]
               for l in out["verdict"]])
    rep.kv("VERDICT", info["VERDICT"])
    rep.write(path)
