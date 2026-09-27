"""Ticket 10 of the Hessian-learning set: the Hessian-vector product by autograd
(`openqha_hessian.hvp`, eq. 9) on a toy potential -- no engine, no MACE weights.

Asserted (the numbers are T03 section 5's, A7): reverse-over-reverse, forward-over-
reverse (`torch.func.jvp` of the gradient) and the explicit `torch.autograd.functional
.hessian` agree to 1e-12; `create_graph=True` returns a tensor whose graph reaches
every parameter that shapes curvature, `create_graph=False` returns one with no
`grad_fn`; a two-structure batch gives each structure's own `H v` (block-diagonal) to
1e-14; k probes come back as [k, n, 3]; a probe with no differentiable path gives
zeros; a malformed probe shape is refused.
"""
import sys

import torch

from openqha_hessian import hvp

FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


class ToyPotential(torch.nn.Module):
    """E(x) = sum_{i<j} f_theta(r_ij) + c sum_i |x_i - x_com|^2 -- T03's toy: a pair MLP
    plus a confining term, parameters theta = (MLP weights, c)."""

    def __init__(self):
        super().__init__()
        self.pair = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        self.conf = torch.nn.Parameter(torch.tensor(0.05))

    def forward(self, x, batch=None):
        n = x.shape[0]
        idx = torch.zeros(n, dtype=torch.long) if batch is None else batch
        same = (idx[:, None] == idx[None, :]).triu(1)
        diff = x[:, None, :] - x[None, :, :]                  # explicit distances: cdist has no double backward
        d = torch.sqrt((diff ** 2).sum(-1)[same] + 1e-12).unsqueeze(-1)
        e_pair = self.pair(d).sum()
        if batch is None:
            com = x.mean(0, keepdim=True)
        else:
            com = torch.stack([x[idx == g].mean(0) for g in idx.unique()])[idx]
        return e_pair + self.conf * ((x - com) ** 2).sum()


def forces_of(model, x, batch=None):
    E = model(x, batch)
    return -torch.autograd.grad(E, x, create_graph=True)[0]


def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    toy = ToyPotential()
    N = 6
    x = torch.randn(N, 3, requires_grad=True)
    v = torch.randn(3, N, 3)                                 # k = 3 probes

    # --- A7: three routes to H v agree ---------------------------------------------------
    F = forces_of(toy, x)
    h_rr = hvp.hvp_from_forces(F, x, v, create_graph=False)
    H = torch.autograd.functional.hessian(lambda p: toy(p), x.detach()).reshape(3 * N, 3 * N)
    h_ref = (v.reshape(3, -1) @ H).reshape(3, N, 3)          # H symmetric: v^T H = (H v)^T
    grad_fn = lambda p: torch.func.grad(lambda q: toy(q))(p)     # noqa: E731
    h_fr = torch.stack([torch.func.jvp(grad_fn, (x.detach(),), (v[j],))[1] for j in range(3)])
    check("shape [k, n, 3]", tuple(h_rr.shape) == (3, N, 3), h_rr.shape)
    check("reverse-over-reverse = explicit H v to 1e-12 (A7)", float((h_rr - h_ref).abs().max()) < 1e-12,
          float((h_rr - h_ref).abs().max()))
    check("forward-over-reverse = explicit H v to 1e-12 (A7)", float((h_fr - h_ref).abs().max()) < 1e-12,
          float((h_fr - h_ref).abs().max()))
    check("create_graph=False: no grad_fn", h_rr.grad_fn is None)

    # --- the graph to the parameters -----------------------------------------------------
    F = forces_of(toy, x)
    h_tr = hvp.hvp_from_forces(F, x, v, create_graph=True)
    check("create_graph=True: has grad_fn", h_tr.grad_fn is not None)
    g = torch.autograd.grad(h_tr.pow(2).sum(), list(toy.parameters()), allow_unused=True)
    names = [n for n, _ in toy.named_parameters()]
    norms = {n: (None if gi is None else float(gi.norm())) for n, gi in zip(names, g)}
    shaping = [n for n in names if n != "pair.2.bias"]        # the last bias shifts E only: no curvature
    check("HVP carries a non-zero gradient to every curvature-shaping parameter",
          all(norms[n] is not None and norms[n] > 0 for n in shaping), norms)
    check("... and none to the energy-only bias", norms["pair.2.bias"] in (None, 0.0), norms["pair.2.bias"])

    # --- batch identity (block-diagonal) -------------------------------------------------
    xa, xb = torch.randn(N, 3), torch.randn(N - 1, 3)
    xab = torch.cat([xa, xb]).requires_grad_(True)
    batch = torch.cat([torch.zeros(N, dtype=torch.long), torch.ones(N - 1, dtype=torch.long)])
    vab = torch.randn(2, 2 * N - 1, 3)
    h_b = hvp.hvp_from_forces(forces_of(toy, xab, batch), xab, vab)
    xa_ = xa.clone().requires_grad_(True); xb_ = xb.clone().requires_grad_(True)
    h_a = hvp.hvp_from_forces(forces_of(toy, xa_), xa_, vab[:, :N])
    h_bb = hvp.hvp_from_forces(forces_of(toy, xb_), xb_, vab[:, N:])
    d = max(float((h_b[:, :N] - h_a).abs().max()), float((h_b[:, N:] - h_bb).abs().max()))
    check("two-structure batch = per-structure HVPs to 1e-14 (block-diagonal)", d < 1e-14, d)

    # --- edges ---------------------------------------------------------------------------
    F = forces_of(toy, x)
    one = hvp.hvp_from_forces(F, x, v[0])                     # a single [n, 3] probe is accepted
    check("a single [n, 3] probe -> [1, n, 3]", tuple(one.shape) == (1, N, 3) and float((one[0] - h_rr[0]).abs().max()) == 0.0)
    try:
        hvp.hvp_from_forces(F, x, torch.randn(2, N + 1, 3))
        check("wrong probe shape refused", False)
    except ValueError as exc:
        check("wrong probe shape refused", "probes must be" in str(exc))
    # a force with no path to positions (a constant): grad is None -> zeros, never None
    const_f = torch.zeros(N, 3, requires_grad=True) * 0 + torch.ones(N, 3)
    z = hvp.hvp_from_forces(const_f, torch.zeros(N, 3, requires_grad=True), v)
    check("no differentiable path -> zeros (allow_unused)", float(z.abs().max()) == 0.0 and tuple(z.shape) == (3, N, 3))

    print("\n{} checks, {} failed".format(12, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
