"""Algorithms 1 and 2 of the PHL loss: the frame's constants, the random probes, the
reference matvec and the exact Cartesian loss (Pengmei, Han, Liu et al., "Probing the
Hessian", eqs. 6 and 2.1').

PRODUCTION. PHL verbatim: one target, the raw Cartesian matrix, and nothing projected
but the probe. numpy only; no mace import.

Everything here is per STRUCTURE and needs no autograd: it is what a data loader or a
loss computes from the Label `H_r` alone, before the model is asked for anything.
Derivations and the numbers the tests hold these functions to:
`docs/tutorials/T04_openQHA_Theory_Hessian_Surface_Learning.ipynb` sections 2 and 6
(Algorithms 1-2) and T05 section 2.

THE TARGET. The raw Cartesian matrix as the reference program wrote
it -- no mass weighting, no Eckart projection, no reference modes, nothing projected but
the random vector itself:

    L_H  = ||H_theta - H_r||_F^2 / (3N)^2                                         (1')
    v_j i.i.d., E[v] = 0, E[v v^T] = I ;   r_j = H_r v_j                          (2')
    L^(K) = sum_j ||H_theta v_j - r_j||^2 / (9 N^2 K)                             (6')

`L^(K)` is unbiased for every K and every unit-variance draw, and the
3N unit probes make it exact -- the deterministic limit of the estimator IS the full
matrix. The production draw is PHL's standard normal (Algorithm 1); the
Rademacher draw has the smaller variance and stays available, but the
target is to follow the published method.

Units: `H` in eV/A^2. The Label is used AS STORED: never symmetrised, projected or
mass-weighted here (the fork's data-loading check refuses a non-symmetric or wrongly
sized Hessian -- that is a data check, not a transformation).

"""
import numpy as np

#: The three probe sets of `make_probes`. `rademacher` and `gaussian` are stochastic
#: (Hutchinson, eq. 6'); `cartesian` is the deterministic unit-vector set that makes the
#: estimator exact at k = 3N HVPs -- the cost of `get_hessian` itself.
PROBE_MODES = ("rademacher", "gaussian", "cartesian")


def loss_full(hessian_theta, hessian_r):
    """The EXACT target, eq. 1': ||H_theta - H_r||_F^2 / (9 N^2) -- the mean squared
    error per matrix element, and the oracle every estimator here is held to."""
    d = np.asarray(hessian_theta, dtype=float) - np.asarray(hessian_r, dtype=float)
    n3 = d.shape[0]
    return float(np.sum(d * d)) / (n3 * n3)


#: Compatibility alias of `loss_full` under the name it carried while two targets
#: existed. Same function, same number.
cartesian_loss_full = loss_full


def make_probes(hessian_r, mode="gaussian", k=4, rng=None):
    """Algorithm 2: the probes the model sees and the reference side of each.

    Returns (v [k, 3N], r [k, 3N], info) with

        v_j           -- the probe, the raw draw: nothing is projected or weighted
        r_j = H_r v_j -- a matvec on the Label (PHL's `bmm(H_ref, v)`)

    and info = dict(n3, mode, k, stochastic, nu, denominator), so that the loss forms
    rho_j = H_theta v_j - r_j and returns sum_j ||rho_j||^2 / denominator. The
    denominator is `nu k` = 9 N^2 k for a STOCHASTIC probe -- every draw estimates the
    whole ||dH||_F^2, so the draws are averaged -- and `nu` = 9 N^2 for the
    DETERMINISTIC set, where probe j is column j of dH and the columns are summed.
    That is the one place the two kinds of probe differ.

    mode: "gaussian" (standard normal, PHL's Algorithm 1 and the default)
    or "rademacher" (v_i in {-1, +1}: the smallest variance of the unit-variance draws
    -- kept as a choice, no longer a default), k draws from `rng` (a numpy
    Generator; the caller seeds it); "cartesian" (v_j = e_j, k := 3N, exact).
    """
    if mode not in PROBE_MODES:
        raise ValueError("probe mode must be one of {}; got {!r}".format(PROBE_MODES, mode))
    hr = np.asarray(hessian_r, dtype=float)
    if hr.ndim != 2 or hr.shape[0] != hr.shape[1]:
        raise ValueError("the Label must be a square [3N, 3N] matrix; got shape {}".format(hr.shape))
    n3 = int(hr.shape[0])
    stochastic = mode in ("rademacher", "gaussian")
    if stochastic:
        if k < 1:
            raise ValueError("k must be >= 1 for a stochastic probe")
        rng = np.random.default_rng() if rng is None else rng
        v = rng.choice([-1.0, 1.0], size=(k, n3)) if mode == "rademacher" else rng.standard_normal((k, n3))
    else:
        v = np.eye(n3)
    v = np.asarray(v, dtype=float)
    r = v @ hr.T                                                      # H_r v_j, row by row
    nu = n3 * n3                                                      # PHL's (3N)^2
    info = dict(n3=n3, mode=mode, k=int(v.shape[0]), stochastic=stochastic, nu=nu,
                denominator=nu * (int(v.shape[0]) if stochastic else 1))
    return v, r, info


def estimator_from_products(hvp_theta, r, info):
    """Eq. 6' from the model's products: sum_j ||H_theta v_j - r_j||^2 over
    `info["denominator"]`, numpy side (the torch side lives in `phl_loss`).
    `hvp_theta` [k, 3N]."""
    hv = np.asarray(hvp_theta, dtype=float).reshape(np.shape(r))
    rho = hv - np.asarray(r, dtype=float)
    return float(np.sum(rho * rho)) / info["denominator"]


def estimator_variance(hessian_theta, hessian_r, k=1):
    """The closed-form variances with A = H_theta - H_r and B = A^T A:

        Var_Rademacher[L^(k)] = 2 (||B||_F^2 - sum_i B_ii^2) / ((9 N^2)^2 k)
        Var_Gaussian  [L^(k)] = 2 ||B||_F^2                  / ((9 N^2)^2 k)

    returned as a dict, the Gaussian one always the larger (the kurtosis term).
    """
    a = np.asarray(hessian_theta, dtype=float) - np.asarray(hessian_r, dtype=float)
    nu = float(a.shape[0]) ** 2                                       # 9 N^2
    b = a.T @ a
    fro2 = float(np.sum(b * b))
    diag2 = float(np.sum(np.diag(b) ** 2))
    return dict(rademacher=2.0 * (fro2 - diag2) / (nu ** 2 * k), gaussian=2.0 * fro2 / (nu ** 2 * k))
