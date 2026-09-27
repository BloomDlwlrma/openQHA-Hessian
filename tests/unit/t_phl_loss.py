"""Tickets 11, 21 and 35 of the Hessian-learning set: the loss module
(`openqha_hessian.phl_loss`) on a toy potential and a stand-in batch carrying ticket
12's fields -- no engine.

Two toy potentials (a pair MLP + confinement) with different parameters play the model
(theta) and the reference; H_r is the reference toy's exact Hessian. There is ONE target
(S0-C-64): the raw Cartesian matrix over (3N)^2.

Asserted -- Algorithm 3 on the loop: the 3N unit probes through the HVP path and the
full-matrix path both equal `phl.loss_full` to 1e-12; autograd dL/dtheta equals a central
finite difference to 1e-6; a two-molecule batch equals the mean of the single-molecule
losses to 1e-13; a third, unlabelled frame changes nothing; a batch with no Label gives 0
and still backpropagates; H_r := H_theta gives 0 with a zero gradient and H_r := 0.81
H_theta gives 0.19^2 ||H_theta||_F^2/(9N^2) with a non-zero one; a `hessian` field of the
wrong length is refused; the Rademacher estimator with k = 4 over 400 seeds has its mean
within 3 sigma of the exact loss and the variance Derivation 2.2 gives.

Algorithm 1 (ticket 35): `FrameConstants` is the Label, nu = 9 N^2 and the frame's seed --
no masses, no positions, no projector, no reference modes; the cache key is the Label's
bytes alone. Algorithm 4 (S0-C-55): in eval mode the Hessian term is the same on two
calls (1e-12, the frame's fixed probes), is the same under another mace seed, differs
between frames, and differs from a training call's fresh draw; `eval_summary()` averages
the three terms over the pass and resets; `wants_hessian_at_eval` is False and
`wants_force_graph_at_eval` True. `build(args)` refuses a target that no longer exists.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from openqha_hessian import phl, phl_loss

FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


class ToyPotential(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.pair = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        self.conf = torch.nn.Parameter(torch.tensor(0.05 + 0.01 * seed))

    def forward(self, x, batch=None):
        n = x.shape[0]
        idx = torch.zeros(n, dtype=torch.long) if batch is None else batch
        same = (idx[:, None] == idx[None, :]).triu(1)
        diff = x[:, None, :] - x[None, :, :]
        d = torch.sqrt((diff ** 2).sum(-1)[same] + 1e-12).unsqueeze(-1)
        e_pair = self.pair(d).sum()
        com = x.mean(0, keepdim=True) if batch is None else torch.stack([x[idx == g].mean(0) for g in idx.unique()])[idx]
        return e_pair + self.conf * ((x - com) ** 2).sum()

    def energies(self, x, batch):
        return torch.stack([self(x[batch == g]) for g in batch.unique()])


class FakeBatch(dict):
    """mace's Batch answers both ref["x"] and ref.x; so does this."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def hessian_of(model, x):
    return torch.autograd.functional.hessian(lambda p: model(p), x.detach()).reshape(3 * x.shape[0], -1).detach().numpy()


K_MAX = 16


def stored_probes(n_atoms, seed, k_max=K_MAX):
    """What the Dataset writes with a labelled frame: [k_max, 3N] ~ N(0, I) (PHL's
    Algorithm 1, S0-C-68) from the frame's IDENTITY (here the caller's `seed` stands for
    it) -- never from the Label (S0-C-67)."""
    return np.random.default_rng(seed).standard_normal((k_max, 3 * n_atoms))


def make_batch(frames, model, probes=None):
    """frames: list of (positions [n,3] numpy, masses [n], H_r [3n,3n] or None). Returns
    (ref, pred) with pred built from `model` with the force graph kept. `probes`: a list
    as long as `frames` of [k_max, 3n] stored sets (None for a frame that carries none);
    by default every labelled frame carries one, seeded by its size, as a valid file's
    frames do."""
    pos = np.vstack([f[0] for f in frames])
    ptr = np.cumsum([0] + [len(f[0]) for f in frames])
    batch = np.concatenate([np.full(len(f[0]), g) for g, f in enumerate(frames)])
    x = torch.tensor(pos, requires_grad=True)
    bt = torch.tensor(batch)
    E = model.energies(x, bt)
    F = -torch.autograd.grad(E.sum(), x, create_graph=True)[0]
    ref = FakeBatch(
        ptr=torch.tensor(ptr), batch=bt, positions=x,
        weight=torch.ones(len(frames)), energy_weight=torch.ones(len(frames)),
        forces_weight=torch.ones(len(frames)), hessian_weight=torch.ones(len(frames)),
        energy=E.detach() + 0.1, forces=F.detach() * 0.9,
        hessian=torch.tensor(np.concatenate([f[2].reshape(-1) for f in frames if f[2] is not None] or [np.zeros(0)])),
        has_hessian=torch.tensor([f[2] is not None for f in frames]),
        sqrt_masses=torch.tensor(np.sqrt(np.concatenate([f[1] for f in frames]))),
    )
    if probes is None:
        probes = [stored_probes(len(f[0]), 700 + len(f[0])) if f[2] is not None else None for f in frames]
    flat = [np.asarray(p, dtype=float).reshape(-1) for p in probes if p is not None]
    ref.valid_probes = torch.tensor(np.concatenate(flat) if flat else np.zeros(0))
    ref.has_valid_probes = torch.tensor([p is not None for p in probes])
    pred = dict(energy=E, forces=F)
    return ref, pred


def full_hessian_pred(model, ref, pred):
    """pred with the full batch Hessian in mace's [3 n_nodes, n_nodes, 3] layout."""
    x = ref["positions"]
    H = torch.autograd.functional.hessian(lambda p: model.energies(p, ref["batch"]).sum(), x.detach())
    n = x.shape[0]
    return dict(pred, hessian=H.reshape(3 * n, n, 3))


def main():
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(3)
    toy = ToyPotential(0)                     # the model
    ref_toy = ToyPotential(5)                 # the reference level
    n_a, n_b = 5, 4
    xa = rng.standard_normal((n_a, 3)) * 1.2
    xb = rng.standard_normal((n_b, 3)) * 1.2
    ma = rng.uniform(1.0, 16.0, n_a)
    mb = rng.uniform(1.0, 16.0, n_b)
    Ha = hessian_of(ref_toy, torch.tensor(xa))
    Hb = hessian_of(ref_toy, torch.tensor(xb))
    Ha_t = hessian_of(toy, torch.tensor(xa))
    Hb_t = hessian_of(toy, torch.tensor(xb))
    exact_a = phl.loss_full(Ha_t, Ha)
    exact_b = phl.loss_full(Hb_t, Hb)

    # --- the 3N unit probes on the loop = the full-matrix path = the numpy exact loss ------
    loss = phl_loss.WeightedEnergyForcesHessianLoss(probe="cartesian")
    ref, pred = make_batch([(xa, ma, Ha)], toy)
    est = loss.hvp_error(ref, pred)
    full = loss.hessian_error_full(ref, full_hessian_pred(toy, ref, pred))
    check("probe=cartesian through the HVP path = ||dH||_F^2/(9N^2) (1e-12)",
          abs(float(est) - exact_a) < 1e-12, (float(est), exact_a))
    check("the full-matrix path = the same number (1e-12)", abs(float(full) - exact_a) < 1e-12, (float(full), exact_a))
    check("the HVP-path term carries a graph; the full-matrix one (detached input) does not need one",
          est.grad_fn is not None)
    total = loss(ref, pred)
    check("forward = w_E L_E + w_F L_F + w_H L_H with mace's E/F terms; last_terms filled",
          abs(float(total) - (loss.last_terms["energy"] + loss.last_terms["forces"] + loss.last_terms["hessian"])) < 1e-12
          and loss.last_terms["n_labelled"] == 1, loss.last_terms)
    pred_h = full_hessian_pred(toy, ref, pred)
    loss(ref, pred_h)
    check("forward with pred['hessian'] takes the exact term (hessian_exact set)",
          loss.last_terms["hessian_exact"] is not None and abs(loss.last_terms["hessian_exact"] - exact_a) < 1e-12)

    # --- the gradient: autograd = finite difference -----------------------------------------
    ref, pred = make_batch([(xa, ma, Ha)], toy)
    L = loss.hvp_error(ref, pred)
    params = list(toy.parameters())
    g = torch.autograd.grad(L, params, allow_unused=True)
    names = [n for n, _ in toy.named_parameters()]
    j0 = names.index("pair.0.weight")          # own parameters come first in parameters(): conf, then the MLP
    p0 = params[j0]                            # first Linear weight [16, 1]
    i = (3, 0); h = 1e-5
    with torch.no_grad():
        orig = float(p0[i])
    def loss_at(val):
        with torch.no_grad():
            p0[i] = val
        r_, pr_ = make_batch([(xa, ma, Ha)], toy)
        v = float(loss.hvp_error(r_, pr_).detach())
        with torch.no_grad():
            p0[i] = orig
        return v
    fd = (loss_at(orig + h) - loss_at(orig - h)) / (2 * h)
    check("autograd dL/dtheta = central finite difference (1e-6 relative)",
          abs(float(g[j0][i]) - fd) < 1e-6 * max(1.0, abs(fd)), (float(g[j0][i]), fd))

    # --- batch identity and masking ---------------------------------------------------------------
    ref2, pred2 = make_batch([(xa, ma, Ha), (xb, mb, Hb)], toy)
    two = float(loss.hvp_error(ref2, pred2))
    check("two-molecule batch = mean of the single-molecule losses (1e-13)",
          abs(two - 0.5 * (exact_a + exact_b)) < 1e-13, (two, exact_a, exact_b))
    ref3, pred3 = make_batch([(xa, ma, Ha), (xb, mb, None), (xb * 1.1, mb, Hb)], toy)
    exact_c = phl.loss_full(hessian_of(toy, torch.tensor(xb * 1.1)), Hb)
    three = float(loss.hvp_error(ref3, pred3))
    check("an unlabelled frame in the batch contributes nothing (mean over the two labelled, 1e-13)",
          abs(three - 0.5 * (exact_a + exact_c)) < 1e-13 and loss.last_terms["n_labelled"] == 2,
          (three, exact_a, exact_c))
    ref0, pred0 = make_batch([(xb, mb, None)], toy)
    z = loss.hvp_error(ref0, pred0)
    z.backward()
    check("a batch with no Label: term 0 and backward() runs", float(z) == 0.0 and loss.last_terms["n_labelled"] == 0)
    bad = FakeBatch(ref); bad["hessian"] = ref["hessian"][:-1]
    try:
        loss.hvp_error(bad, pred)
        check("a hessian field of the wrong length is refused", False)
    except ValueError as exc:
        check("a hessian field of the wrong length is refused", "batch.hessian holds" in str(exc))

    # --- the two limits through the module -------------------------------------------------------------
    ref5, pred5 = make_batch([(xa, ma, Ha_t)], toy)
    L5 = loss.hvp_error(ref5, pred5)
    g5 = torch.autograd.grad(L5, params, allow_unused=True)
    check("must-pass: H_r := H_theta -> 0 and zero gradient (1e-16)",
          float(L5) < 1e-16 and max(float(x.abs().max()) for x in g5 if x is not None) < 1e-10, float(L5))
    ref6, pred6 = make_batch([(xa, ma, 0.81 * Ha_t)], toy)
    L6 = loss.hvp_error(ref6, pred6)
    g6 = torch.autograd.grad(L6, params, allow_unused=True)
    a6 = 0.19 ** 2 * float(np.sum(Ha_t * Ha_t)) / (3 * n_a) ** 2
    check("must-fail: H_r := 0.81 H_theta -> 0.19^2 ||H_theta||_F^2/(9N^2) and a non-zero gradient",
          abs(float(L6) / a6 - 1) < 1e-10 and max(float(x.abs().max()) for x in g6 if x is not None) > 1e-6,
          (float(L6), a6))

    # --- Rademacher k = 4 on the loop -------------------------------------------------------------------
    var = phl.estimator_variance(Ha_t, Ha, k=4)["rademacher"]
    n_seeds = 400
    vals = []
    for s in range(n_seeds):
        l4 = phl_loss.WeightedEnergyForcesHessianLoss(probe="rademacher", n_probes=4, seed=s)
        r_, pr_ = make_batch([(xa, ma, Ha)], toy)
        vals.append(float(l4.hvp_error(r_, pr_).detach()))
    vals = np.array(vals)
    sig_mean = np.sqrt(var / n_seeds)
    check("Rademacher k=4 over {} seeds: mean within 3 sigma of the exact loss ({:.2f} sigma), "
          "sample variance within 25 % of Derivation 2.2".format(n_seeds, abs(vals.mean() - exact_a) / sig_mean),
          abs(vals.mean() - exact_a) < 3 * sig_mean and abs(vals.var() / var - 1) < 0.25,
          (vals.mean(), exact_a, vals.var(), var))

    # --- Algorithm 1: what the frame's constants are now (ticket 35) --------------------------------------
    c = loss.constants(Ha)
    check("FrameConstants = the Label and nu = 9 N^2 -- three slots, nothing else (S0-C-67)",
          c.n3 == 3 * n_a and c.nu == (3 * n_a) ** 2
          and set(phl_loss.FrameConstants.__slots__) == {"hessian_r", "n3", "nu"},
          phl_loss.FrameConstants.__slots__)
    check("no seed is derived from the Label anywhere: frame_seed is gone and phl_loss imports no hashlib",
          not hasattr(phl_loss, "frame_seed") and not hasattr(c, "seed")
          and "hashlib" not in Path(phl_loss.__file__).read_text(encoding="utf-8"))
    for gone in ("masses", "positions", "metric", "modes_r", "lam_r", "weights", "projector",
                 "inv_sqrt_m", "n_vib", "projector_t", "inv_sqrt_m_t"):
        if hasattr(c, gone):
            check("FrameConstants no longer carries {!r}".format(gone), False)
    check("FrameConstants carries none of the eleven projected attributes", True)
    check("the constants are built per call and keyed by nothing (the cache went with the hash)",
          loss.constants(Ha) is not c and loss.constants(Ha).nu == c.nu)
    try:
        loss.constants(Ha, ma, xa)
        check("constants(H_r) takes the Label alone", False)
    except TypeError:
        check("constants(H_r) takes the Label alone", True)

    # --- build(args) and the construction refusals ------------------------------------------------------
    args = argparse.Namespace(energy_weight=1.0, forces_weight=100.0, hessian_weight=7.5, n_hessian_probes=2,
                              hessian_probe="gaussian", seed=11)
    b = phl_loss.build(args)
    check("build(args) returns the class with the flags",
          isinstance(b, phl_loss.WeightedEnergyForcesHessianLoss) and float(b.hessian_weight) == 7.5 and b.n_probes == 2
          and b.probe == "gaussian" and b.seed == 11 and float(b.forces_weight) == 100.0)
    check("repr names every setting and the one target",
          "hessian_weight=7.500" in repr(b) and "probe='gaussian'" in repr(b) and "seed=11" in repr(b)
          and "target='cartesian'" in repr(b), repr(b))
    check("wants_hessian_at_eval is False and wants_force_graph_at_eval True (S0-C-55; the fork's evaluate reads both)",
          b.wants_hessian_at_eval is False and b.wants_force_graph_at_eval is True)
    b_default = phl_loss.build(argparse.Namespace(energy_weight=1.0, forces_weight=100.0, hessian_weight=1.0, seed=1))
    check("build(args) without any target flag builds the Cartesian loss",
          b_default.probe == "gaussian" and b_default.n_probes == 4 and "target='cartesian'" in repr(b_default))
    b_stale = phl_loss.build(argparse.Namespace(energy_weight=1.0, forces_weight=1.0, hessian_weight=1.0,
                                               seed=1, hessian_mode_weighting="entropy"))
    check("build(args) ignores a stale hessian_mode_weighting: the flag is out of the fork's parser "
          "(commit D), so mace itself refuses it and nothing here can receive it",
          "target='cartesian'" in repr(b_stale))
    check("MODE_WEIGHTINGS and DEFAULT_MODE_WEIGHTING are gone from the module",
          not hasattr(phl_loss, "MODE_WEIGHTINGS") and not hasattr(phl_loss, "DEFAULT_MODE_WEIGHTING"))
    for gone in ("hutchinson", "modes"):
        try:
            phl_loss.WeightedEnergyForcesHessianLoss(probe=gone)
            check("probe={!r} is refused at construction".format(gone), False)
        except ValueError:
            check("probe={!r} is refused at construction".format(gone), True)
    try:
        phl_loss.WeightedEnergyForcesHessianLoss(mode_weighting="none")
        check("the constructor takes no `mode_weighting` (there is one target)", False)
    except TypeError:
        check("the constructor takes no `mode_weighting` (there is one target)", True)

    # --- validation: four probes fixed per frame (S0-C-55) ---------------------------------------------
    lv = phl_loss.WeightedEnergyForcesHessianLoss(probe="rademacher", n_probes=4, seed=3)
    lv.eval()
    ref, pred = make_batch([(xa, ma, Ha)], toy)
    v1 = float(lv.hvp_error(ref, pred))
    ref, pred = make_batch([(xa, ma, Ha)], toy)
    v2 = float(lv.hvp_error(ref, pred))
    check("eval mode: two calls on the same frame give the same Hessian term (1e-12) -- the probes are fixed",
          abs(v1 - v2) < 1e-12 and v1 > 0, (v1, v2))
    lv2 = phl_loss.WeightedEnergyForcesHessianLoss(probe="rademacher", n_probes=4, seed=99)
    lv2.eval()
    ref, pred = make_batch([(xa, ma, Ha)], toy)
    check("... and a second module with another mace seed gives the same value: the probes come from the file, not the seed",
          abs(float(lv2.hvp_error(ref, pred)) - v1) < 1e-12)

    # S0-C-67: the probes do not move when the Label is rewritten bit-for-bit differently at
    # the SAME level -- this is the regression the stored set exists for
    Ha_rewritten = Ha + 0.0                      # a different object, the same physics
    Ha_rewritten[0, 0] = np.nextafter(Ha[0, 0], np.inf)
    ref, pred = make_batch([(xa, ma, Ha_rewritten)], toy)
    check("a Label recomputed at the same level leaves the probes untouched (the old SHA-1 seed "
          "would have changed every one of them)",
          abs(float(lv.hvp_error(ref, pred)) - v1) < 1e-9)

    store_a = stored_probes(n_a, 700 + n_a)
    vt_a, _r, _i = lv.constants(Ha).probes("rademacher", 4, stored=store_a)
    vt_8, _r8, _i8 = lv.constants(Ha).probes("rademacher", 8, stored=store_a)
    check("the K = 4 set is the first four rows of the stored set, and K = 8 is nested with it",
          vt_a.shape == (4, 3 * n_a) and np.array_equal(vt_8[:4], vt_a) and vt_8.shape == (8, 3 * n_a))
    store_b = stored_probes(n_b, 700 + n_b)
    vt_b, _r, _i = lv.constants(Hb).probes("rademacher", 4, stored=store_b)
    check("different frames get different fixed probes (the Dataset drew them from different identities)",
          vt_b.shape == (4, 3 * n_b) and not np.array_equal(vt_a[:, :12], vt_b[:, :12]))
    try:
        lv.constants(Ha).probes("rademacher", 4)
        check("a labelled validation frame with no stored probes is REFUSED, not drawn for", False)
    except ValueError as exc:
        check("a labelled validation frame with no stored probes is REFUSED, not drawn for",
              "S0-C-67" in str(exc) and "04_dataset" in str(exc), str(exc))
    try:
        lv.constants(Ha).probes("rademacher", 32, stored=store_a)
        check("asking for more probes than the Dataset stored is refused naming VALID_PROBE_KMAX", False)
    except ValueError as exc:
        check("asking for more probes than the Dataset stored is refused naming VALID_PROBE_KMAX",
              "VALID_PROBE_KMAX" in str(exc), str(exc))
    ref_no, pred_no = make_batch([(xa, ma, Ha)], toy, probes=[None])
    try:
        lv.hvp_error(ref_no, pred_no)
        check("... and the same refusal reaches the loss through a stale valid file", False)
    except ValueError as exc:
        check("... and the same refusal reaches the loss through a stale valid file", "valid_probes" in str(exc))
    lv.train()
    ref, pred = make_batch([(xa, ma, Ha)], toy)
    t1 = float(lv.hvp_error(ref, pred))
    ref, pred = make_batch([(xa, ma, Ha)], toy)
    t2 = float(lv.hvp_error(ref, pred))
    check("training mode: fresh draws, two calls differ", abs(t1 - t2) > 1e-9, (t1, t2))
    var_c = phl.estimator_variance(Ha_t, Ha, k=4)["rademacher"]
    check("the fixed validation value is within 4 sigma of the exact loss (Derivation 2.2 for k = 4)",
          abs(v1 - exact_a) < 4 * np.sqrt(var_c), (v1, exact_a, np.sqrt(var_c)))
    # the summary over a pass: E and F averaged per graph, H per labelled graph, then reset
    lv.eval()
    ref, pred = make_batch([(xa, ma, Ha), (xb, mb, None)], toy)
    lv(ref, pred)
    ref, pred = make_batch([(xb, mb, Hb)], toy)
    lv(ref, pred)
    summary = lv.eval_summary()
    hb_fixed = float(lv.hvp_error(*make_batch([(xb, mb, Hb)], toy)))
    check("eval_summary: three terms over the pass (H = mean of the two labelled graphs' fixed values), "
          "n_labelled 2, the label of the probes; then reset",
          abs(summary["valid_hessian_term"] - 0.5 * (v1 + hb_fixed)) < 1e-12 and summary["valid_hessian_n_labelled"] == 2
          and summary["valid_probes"] == phl_loss.VALID_PROBES_LABEL
          and summary["valid_forces_term"] is not None and summary["valid_energy_term"] is not None
          and lv._eval_sums["n_batches"] == 0,           # reset; hvp_error alone (above) does not accumulate
          summary)
    check("eval_summary no longer reports a target: there is one (S0-C-64)", "valid_target" not in summary, summary)
    lv.eval_summary()
    check("eval_summary on an empty pass answers None terms", lv.eval_summary()["valid_hessian_term"] is None)
    lv.train()
    ref, pred = make_batch([(xa, ma, Ha)], toy)
    lv(ref, pred)
    check("training-mode forwards do not accumulate", lv._eval_sums["n_batches"] == 0)

    print("\n{} checks, {} failed".format(41, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
