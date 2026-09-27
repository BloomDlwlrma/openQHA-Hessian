"""The Hessian loss as a mace loss module: Algorithm 3 with the HVP estimator (eq. 6')
in training and the SAME estimator on four probes fixed per frame in evaluation.

PRODUCTION. Ticket 11 of the Hessian-learning set; the Cartesian target and the
fixed-probe validation of ticket 21 (S0-C-53, S0-C-55); PHL verbatim by ticket 35
(S0-C-64, 2026-09-23): there is ONE target and nothing is projected but the probe.

This is the `nn.Module` mace's `train()` calls as `loss_fn(pred=output, ref=batch)`. It
reaches mace through the fork's external-loss hook (ticket 13: `--loss external
--loss_module openqha_hessian.phl_loss:build`); the E and F terms are mace's own
(`mace.modules.loss`, public), the Hessian term is here. The batch fields it reads are
the ones the fork's commit A adds (ticket 12): `hessian` (the Labels, flattened 9 n_k^2
per graph and concatenated like `ptr`), `has_hessian` [n_graphs], `hessian_weight`
[n_graphs], `sqrt_masses` [n_nodes]; and mace's `ptr`, `weight`, `positions`.

THE TARGET. The raw Cartesian matrix, PHL's eq. 2.1': rho_j = H_theta v_j - H_r v_j
over 9 N^2 k -- no mass weighting, no Eckart projection, no reference modes, no
entropy weights. The projected targets of T03 were deleted with ticket 35; the standard
vibrational analysis is the judge's business, not the loss's.

TRAINING. The HVP is taken INSIDE the loss from `pred["forces"]` and `ref["positions"]`
(`hvp.hvp_from_forces`, create_graph=True) -- one backward pass per probe index for the
whole batch, since a batch is block-diagonal. The probes are fresh draws from the
module's generator (mace's seed) every step.

EVALUATION (S0-C-55). The fork's `evaluate` puts the loss in eval mode (commit C) and,
because `wants_force_graph_at_eval` is set, calls the model with the force graph kept;
the Hessian term is then the same estimator on `VALID_N_PROBES` standard-normal probes that
the DATASET drew and stored with the frame (S0-C-67, PHL's fixed-vector protocol): the
loss takes the first k rows of `ref.valid_probes` and draws nothing. They are identical
every epoch for a frame, independent between frames, reproducible from the Dataset's
Record alone, and unchanged when the Label is recomputed at the same level -- so the
validation curve is a fixed, cheap, unbiased reading of the target (about 4 HVPs per
labelled frame against 3N for the full matrix). A labelled validation frame that carries
no stored set is REFUSED, not drawn for: that state means a stale valid file. That
term enters the total validation loss mace's scheduler, checkpoint and Stage Two read.
Over a validation pass the three terms are accumulated and handed back through
`eval_summary()` (commit C), which joins them to mace's `results/*.txt` as
`valid_energy_term`, `valid_forces_term`, `valid_hessian_term`. `wants_hessian_at_eval`
is False: the fork's full-matrix hook stays available but unused; the full matrix is
the judge's tool (`hessian_error_full` remains for it and for the tests).

Per-structure constants are the Label and one number (Algorithm 1): nu = 9 N^2. There is
no seed, no cache and no hash in this module -- the probes ride in with the graph, so
nothing has to be derived from the Label's bytes. Nothing is diagonalised.
"""
import logging

import numpy as np
import torch

from openqha_hessian import hvp as hvp_mod
from . import phl

#: the validation estimator (S0-C-55, S0-C-67, S0-C-68): k fixed standard-normal probes per
#: frame, drawn by the Dataset and read from the file
VALID_PROBE = "gaussian"                                   # PHL's Algorithm 1 (S0-C-68)
VALID_N_PROBES = 4
PROBES_LABEL_FORM = "{} k={} fixed, stored by the Dataset"
VALID_PROBES_LABEL = PROBES_LABEL_FORM.format(VALID_PROBE, VALID_N_PROBES)


def _field(ref, name):
    """A batch field, by item or attribute (mace's Batch answers both; a test's stand-in
    may answer one)."""
    try:
        return ref[name]
    except (TypeError, KeyError, IndexError):
        return getattr(ref, name)


def _has(ref, name):
    try:
        return _field(ref, name) is not None
    except (AttributeError, KeyError):
        return False


class FrameConstants:
    """Algorithm 1: what the loss needs of one frame -- the Label as stored and nu = 9 N^2.
    Nothing is diagonalised, projected, mass-weighted (S0-C-64) or hashed (S0-C-67)."""

    __slots__ = ("hessian_r", "n3", "nu")

    def __init__(self, hessian_r):
        self.hessian_r = np.asarray(hessian_r, dtype=float)
        self.n3 = int(self.hessian_r.shape[0])
        self.nu = self.n3 * self.n3                                    # PHL's (3N)^2

    def probes(self, mode, k, rng=None, stored=None):
        """Algorithm 2's probes for this frame.

        TRAINING (`rng` given): a fresh draw every step, from mace's generator.
        VALIDATION (`rng` None): the frame's STORED set -- the first k of the
        [k_max, 3N] rows the Dataset drew from the frame's identity and wrote into the
        valid file (S0-C-67). The loss draws nothing there and derives nothing from the
        Label: a Label recomputed at the same level leaves these vectors untouched.
        """
        if rng is not None:
            return phl.make_probes(self.hessian_r, mode=mode, k=k, rng=rng)
        if stored is None:
            raise ValueError(
                "a labelled validation frame carries no valid_probes: the fixed probes are drawn by "
                "the Dataset and stored in the valid file (S0-C-67), and this loss draws none. "
                "Rebuild the Dataset with 04_dataset.py so that valid.<level>.extxyz carries "
                "REF_valid_probes, and train against a fork that has commit D.")
        v = np.asarray(stored, dtype=float)
        if v.ndim != 2 or v.shape[1] != self.n3:
            raise ValueError("valid_probes for this frame is {}, not [k_max, {}]".format(v.shape, self.n3))
        if k > v.shape[0]:
            raise ValueError(
                "the validation asks for {} probes and the Dataset stored {} per frame (VALID_PROBE_KMAX); "
                "rebuild the Dataset to store more, or lower --valid-n-probes".format(k, v.shape[0]))
        v = v[:int(k)]
        r = v @ self.hessian_r.T
        return v, r, dict(n3=self.n3, mode=mode, k=int(k), stochastic=True, nu=self.nu,
                          denominator=self.nu * int(k))


def graph_labels(ref):
    """Per graph: (index, n_atoms, H_r [3n, 3n] numpy or None, masses, positions, probes)
    read from the batch fields of tickets 12 and 36. Graphs without a Label give None for
    `H_r`; graphs without a stored probe set give None for `probes` (every labelled frame has
    one, S0-C-67 -- mace evaluates the loss on the training split too; a pool frame, which
    has no Label at all, has none)."""
    ptr = _field(ref, "ptr").detach().cpu().numpy()
    n_k = ptr[1:] - ptr[:-1]
    has = _field(ref, "has_hessian").detach().cpu().numpy().astype(bool) if _has(ref, "has_hessian") \
        else np.ones(len(n_k), dtype=bool)
    hess = _field(ref, "hessian").detach().cpu().numpy() if _has(ref, "hessian") else np.zeros(0)
    sqrt_m = _field(ref, "sqrt_masses").detach().cpu().numpy()
    pos = _field(ref, "positions").detach().cpu().numpy()
    has_v = _field(ref, "has_valid_probes").detach().cpu().numpy().astype(bool) \
        if _has(ref, "has_valid_probes") else np.zeros(len(n_k), dtype=bool)
    vprobes = _field(ref, "valid_probes").detach().cpu().numpy() if _has(ref, "valid_probes") else np.zeros(0)
    # k_max is one number for the whole Dataset, so it is read once from what the batch holds
    k_max = _k_max_from(vprobes, has_v, n_k) if vprobes.size else 0
    out, off, off_v = [], 0, 0
    for g, n in enumerate(n_k):
        size = 9 * int(n) ** 2 if has[g] else 0
        h = None
        if has[g]:
            if off + size > hess.size:
                raise ValueError("batch.hessian holds {} numbers; graph {} needs {} more at offset {}".format(
                    hess.size, g, size, off))
            h = hess[off:off + size].reshape(3 * n, 3 * n)
        off += size
        v = None
        if has_v[g] and k_max:
            n3 = 3 * int(n)
            size_v = k_max * n3
            if off_v + size_v > vprobes.size:
                raise ValueError("batch.valid_probes holds {} numbers; graph {} needs {} more at offset {}".format(
                    vprobes.size, g, size_v, off_v))
            v = vprobes[off_v:off_v + size_v].reshape(k_max, n3)
            off_v += size_v
        a, b = int(ptr[g]), int(ptr[g + 1])
        out.append((g, int(n), h, sqrt_m[a:b] ** 2, pos[a:b], v))
    if off != hess.size:
        raise ValueError("batch.hessian holds {} numbers but the labelled graphs account for {}".format(hess.size, off))
    if off_v != vprobes.size:
        raise ValueError("batch.valid_probes holds {} numbers but the graphs account for {}".format(vprobes.size, off_v))
    return out


def _k_max_from(vprobes, has_v, n_k):
    """k_max = (total stored numbers) / (sum of 3n over the graphs that stored a set): one
    number for the whole batch, because a Dataset stores the same k_max for every frame."""
    denom = sum(3 * int(n) for g, n in enumerate(n_k) if has_v[g])
    if denom == 0 or vprobes.size % denom != 0:
        raise ValueError("batch.valid_probes holds {} numbers, not a multiple of sum(3n) = {} over the "
                         "graphs that carry a set".format(vprobes.size, denom))
    return vprobes.size // denom


class WeightedEnergyForcesHessianLoss(torch.nn.Module):
    """w_E L_E + w_F L_F + w_H L_H (Algorithm 3): mace's weighted E and F terms plus the
    Hessian term -- the estimator (eq. 6') from the forces' graph when `pred` has no
    "hessian" (training: fresh probes; evaluation: the frame's fixed probes), the exact
    full-matrix value (eq. 1') when it has (the judge's path)."""

    #: the fork's `evaluate` reads these: no full Hessian at evaluation (S0-C-55), but
    #: the force graph kept so the estimator can take its HVPs there
    wants_hessian_at_eval = False
    wants_force_graph_at_eval = True

    def __init__(self, energy_weight=1.0, forces_weight=1.0, hessian_weight=1.0, n_probes=4,
                 probe="gaussian", seed=None, cache_size=4096,
                 valid_probe=VALID_PROBE, valid_n_probes=VALID_N_PROBES):
        super().__init__()
        if probe not in phl.PROBE_MODES:
            raise ValueError("probe must be one of {}; got {!r}".format(phl.PROBE_MODES, probe))
        dt = torch.get_default_dtype()
        self.register_buffer("energy_weight", torch.tensor(float(energy_weight), dtype=dt))
        self.register_buffer("forces_weight", torch.tensor(float(forces_weight), dtype=dt))
        self.register_buffer("hessian_weight", torch.tensor(float(hessian_weight), dtype=dt))
        self.n_probes = int(n_probes)
        self.probe = probe
        self.seed = seed
        self.valid_probe = valid_probe
        self.valid_n_probes = int(valid_n_probes)
        self.rng = np.random.default_rng(seed)
        self._cache = {}
        self._cache_size = int(cache_size)
        # the last values of the three terms, for the training log
        self.last_terms = dict(energy=None, forces=None, hessian=None, hessian_exact=None, n_labelled=0)
        self._eval_sums = None
        self._reset_eval_sums()

    # ---- per-frame constants ---------------------------------------------------------
    def constants(self, hessian_r):
        """Algorithm 1 for one frame. Not cached and not keyed: since ticket 35 a
        FrameConstants is an `asarray` and two integers, and since S0-C-67 there is
        nothing left that a cache key could be derived from -- the probes come with the
        graph, not from the Label."""
        return FrameConstants(hessian_r)

    # ---- Algorithm 2 for a batch -----------------------------------------------------
    def make_probes(self, ref, like):
        """Probes for the whole batch: a tensor [k_max, n_nodes, 3] (zero rows for
        graphs without a Label and for j >= k_g), and per labelled graph
        (g, slice, k_g, v [k_g, 3n], r_j [k_g, 3n] numpy, constants, denominator).
        In training mode the probes are fresh draws (`self.probe`, `self.n_probes`); in
        eval mode the first `valid_n_probes` rows of the set the Dataset stored with the
        frame (S0-C-67) -- nothing is drawn and nothing is derived from the Label."""
        labels = graph_labels(ref)
        ptr = _field(ref, "ptr").detach().cpu().numpy()
        n_nodes = int(ptr[-1])
        per_graph = []
        k_max = 0
        for g, n, h_r, _masses, _pos, stored in labels:
            if h_r is None:
                continue
            c = self.constants(h_r)
            if self.training:
                vt, r, info = c.probes(self.probe, self.n_probes, rng=self.rng)
            else:
                vt, r, info = c.probes(self.valid_probe, self.valid_n_probes, stored=stored)
            per_graph.append((g, slice(int(ptr[g]), int(ptr[g + 1])), info["k"], vt, r, c, info["denominator"]))
            k_max = max(k_max, info["k"])
        probes = torch.zeros((k_max, n_nodes, 3), dtype=like.dtype, device=like.device)
        for g, sl, k_g, vt, _r, _c, _den in per_graph:
            probes[:k_g, sl, :] = torch.as_tensor(vt.reshape(k_g, -1, 3), dtype=like.dtype, device=like.device)
        return probes, per_graph

    # ---- the two Hessian terms -----------------------------------------------------------
    def hvp_error(self, ref, pred):
        """Eq. 6' per labelled graph, weighted by weight * hessian_weight, averaged over
        the labelled graphs; 0 (with a graph) when the batch has no Label."""
        forces = pred["forces"]
        positions = _field(ref, "positions")
        probes, per_graph = self.make_probes(ref, forces)
        if not per_graph:
            self.last_terms.update(hessian=0.0, n_labelled=0)
            return 0.0 * forces.sum()
        if forces.grad_fn is None:
            raise RuntimeError(
                "the forces carry no graph, so no Hessian-vector product can be taken: the model must be "
                "called with training=True (the fork's evaluate does so when the loss sets "
                "wants_force_graph_at_eval, commit C)")
        # the third-order graph (to the parameters) only in training; a validation HVP is a value.
        # torchmetrics runs a metric's update under no_grad (mace's evaluate wraps the loss in
        # one), so grad is re-enabled here: the force graph exists, it only has to be walked
        with torch.enable_grad():
            hv = hvp_mod.hvp_from_forces(forces, positions, probes, create_graph=bool(self.training))
        if not self.training:
            hv = hv.detach()
        w_cfg = _field(ref, "weight")
        w_h = _field(ref, "hessian_weight") if _has(ref, "hessian_weight") else torch.ones_like(w_cfg)
        terms = []
        for g, sl, k_g, _v, r, _c, den in per_graph:
            hv_g = hv[:k_g, sl, :].reshape(k_g, -1)                                  # [k_g, 3n]
            rho = hv_g - torch.as_tensor(r, dtype=forces.dtype, device=forces.device)
            terms.append(w_cfg[g] * w_h[g] * (rho * rho).sum() / den)                # 9 N^2 k (stochastic) or 9 N^2
        raw = torch.stack(terms)
        self.last_terms.update(hessian=float(raw.detach().mean()), n_labelled=len(terms))
        return raw.mean()

    def hessian_error_full(self, ref, pred):
        """Eq. 1' per labelled graph from the model's full Hessian `pred["hessian"]`
        ([3 n_nodes, n_nodes, 3] as mace's `compute_hessians_vmap` returns it for a
        batch, or [3n, 3n] for one graph), same weighting and mean."""
        h_all = pred["hessian"]
        ptr = _field(ref, "ptr").detach().cpu().numpy()
        n_nodes = int(ptr[-1])
        h_all = h_all.reshape(3 * n_nodes, n_nodes, 3)
        w_cfg = _field(ref, "weight")
        w_h = _field(ref, "hessian_weight") if _has(ref, "hessian_weight") else torch.ones_like(w_cfg)
        terms = []
        for g, n, h_r, _masses, _pos, _probes in graph_labels(ref):
            if h_r is None:
                continue
            c = self.constants(h_r)
            a, b = int(ptr[g]), int(ptr[g + 1])
            h_t = h_all[3 * a:3 * b, a:b, :].reshape(3 * n, 3 * n)
            d = h_t - torch.as_tensor(h_r, dtype=h_t.dtype, device=h_t.device)
            terms.append(w_cfg[g] * w_h[g] * (d * d).sum() / c.nu)                   # 9 N^2
        if not terms:
            self.last_terms.update(hessian_exact=0.0, n_labelled=0)
            return 0.0 * h_all.sum()
        raw = torch.stack(terms)
        self.last_terms.update(hessian_exact=float(raw.detach().mean()), n_labelled=len(terms))
        return raw.mean()

    # ---- the validation pass ---------------------------------------------------------------
    def _reset_eval_sums(self):
        self._eval_sums = dict(energy=0.0, forces=0.0, hessian=0.0, n_graphs=0, n_labelled=0, n_batches=0,
                               last_batch=None)

    def _accumulate(self, ref, loss_e, loss_f, hessian_mean, n_labelled):
        s = self._eval_sums
        # torchmetrics' full-state update calls the loss twice on the same batch object (the
        # global and the per-batch state); the second call carries the same numbers and
        # would only double the counts, so a batch is accumulated once
        if s["last_batch"] is not None and s["last_batch"] is ref:
            return
        s["last_batch"] = ref
        n_graphs = int(_field(ref, "ptr").numel() - 1)
        s["energy"] += float(loss_e.detach()) * n_graphs
        s["forces"] += float(loss_f.detach()) * n_graphs
        s["hessian"] += float(hessian_mean) * n_labelled
        s["n_graphs"] += n_graphs
        s["n_labelled"] += n_labelled
        s["n_batches"] += 1

    def eval_summary(self):
        """The three terms averaged over the validation pass (E and F per graph, the
        Hessian per labelled graph), as the fork's `evaluate` merges them into the
        metrics it logs; resets the accumulators. Called once per validation pass."""
        s = self._eval_sums
        out = dict(valid_energy_term=(s["energy"] / s["n_graphs"]) if s["n_graphs"] else None,
                   valid_forces_term=(s["forces"] / s["n_graphs"]) if s["n_graphs"] else None,
                   valid_hessian_term=(s["hessian"] / s["n_labelled"]) if s["n_labelled"] else None,
                   valid_hessian_n_labelled=int(s["n_labelled"]),
                   valid_probes=PROBES_LABEL_FORM.format(self.valid_probe, self.valid_n_probes))
        if s["n_batches"]:
            logging.info("openQHA loss (valid): energy=%s forces=%s hessian=%s n_labelled=%d probes=%s",
                         "-" if out["valid_energy_term"] is None else "{:.6e}".format(out["valid_energy_term"]),
                         "-" if out["valid_forces_term"] is None else "{:.6e}".format(out["valid_forces_term"]),
                         "-" if out["valid_hessian_term"] is None else "{:.6e}".format(out["valid_hessian_term"]),
                         out["valid_hessian_n_labelled"], out["valid_probes"])
        self._reset_eval_sums()
        return out

    # ---- eq. 11 --------------------------------------------------------------------------
    def forward(self, ref, pred, ddp=None):
        from mace.modules.loss import mean_squared_error_forces, weighted_mean_squared_error_energy
        loss_e = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_f = mean_squared_error_forces(ref, pred, ddp)
        if "hessian" in pred and pred["hessian"] is not None:
            loss_h = self.hessian_error_full(ref, pred)
            h_value = self.last_terms["hessian_exact"]
        else:
            loss_h = self.hvp_error(ref, pred)
            h_value = self.last_terms["hessian"]
        self.last_terms.update(energy=float(loss_e.detach()), forces=float(loss_f.detach()))
        if not self.training:
            self._accumulate(ref, loss_e, loss_f, h_value or 0.0, self.last_terms["n_labelled"])
        return self.energy_weight * loss_e + self.forces_weight * loss_f + self.hessian_weight * loss_h

    def __repr__(self):
        return ("{}(energy_weight={:.3f}, forces_weight={:.3f}, hessian_weight={:.3f}, "
                "n_probes={}, probe={!r}, target='cartesian', seed={!r}, valid_probes={!r})").format(
                    self.__class__.__name__, float(self.energy_weight), float(self.forces_weight),
                    float(self.hessian_weight), self.n_probes, self.probe, self.seed,
                    PROBES_LABEL_FORM.format(self.valid_probe, self.valid_n_probes))


def build(args):
    """The factory the fork's `--loss external --loss_module openqha_hessian.phl_loss:build`
    calls with mace's parsed arguments (ticket 13's flags; every one has a default).

    There is one target (S0-C-64), so there is nothing here to select it with: fork
    commit D took `--hessian_mode_weighting` and `--hessian_probe modes` out of the
    parser (ticket 36), and a run that names either now fails in mace's own argument
    parsing, before a model is built.
    """
    return WeightedEnergyForcesHessianLoss(
        energy_weight=getattr(args, "energy_weight", 1.0),
        forces_weight=getattr(args, "forces_weight", 1.0),
        hessian_weight=getattr(args, "hessian_weight", 1.0),
        n_probes=getattr(args, "n_hessian_probes", 4),
        probe=getattr(args, "hessian_probe", "gaussian"),
        seed=getattr(args, "seed", None),
    )
