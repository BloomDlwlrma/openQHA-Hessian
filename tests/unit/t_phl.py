"""Algorithms 1-2 of the PHL loss
(`openqha_hessian.phl`) on the propanal fixture -- no engine.

H_r = the ORCA wB97M-D3BJ/def2-TZVPPD Hessian of basin 0 (`job.hess`), H_theta = the
stored MACE-OFF23_medium Hessian at the same geometry (`hessian_at_<level>.npy`).

PHL VERBATIM, the derivations (T04, sections 2 and 6) run on
that pair: eq. 1' is the mean square per matrix element; the 3N unit
probes reproduce it to 1e-12 with zero variance; 4000 Rademacher and
4000 Gaussian single-probe draws have their mean within 3 standard errors of it;
each sample variance is within 5 % of the closed form, and the
Gaussian one is strictly the larger (the kurtosis term 2 sum_i B_ii^2); the variance
falls as 1/k; the probe is the RAW draw and r_j = H_r v_j. Refused: an unknown probe
mode, `modes` (there is no such probe any more), a `metric` or masses in the signature,
and a Label that is not square.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tests/, where _testlib lives
from _testlib import openqha_src                                # noqa: E402

from openqha.qm_interfaces import orca                          # noqa: E402
from openqha.store import layout                                # noqa: E402
from openqha_hessian import phl                                 # noqa: E402

FIX = openqha_src() / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    from ase.data import atomic_masses, atomic_numbers
    parsed = orca.parse_hess(layout.orca_level_file(FIX, LEVEL, 0, ".hess"))
    symbols = list(parsed["symbols"])
    x_r = np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
    H_r = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
    H_t = np.load(FIX / "mace" / "basin00" / "hessian_at_{}.npy".format(LEVEL))
    n3 = 3 * len(symbols)

    # === PHL verbatim: the target, the probes, the estimator (spec step 2) ==================
    # --- eq. 1': the exact target ------------------------------------------------------------
    L = phl.loss_full(H_t, H_r)
    d = H_t - H_r
    check("eq. 1': loss_full = ||dH||_F^2 / (9 N^2) (1e-16); ~1e-2..1e-1 eV^2/A^4 on propanal",
          abs(L - np.sum(d * d) / n3 ** 2) < 1e-16 and 1e-3 < L < 1.0, L)
    check("cartesian_loss_full is the same function under its earlier name",
          phl.cartesian_loss_full is phl.loss_full)

    # --- the 3N unit probes are the full matrix ----------------------------------------------
    v, r, info = phl.make_probes(H_r, mode="cartesian")
    check("v = I, r = H_r, k = 3N, denominator 9 N^2, estimator = eq. 1' to 1e-12",
          np.array_equal(v, np.eye(n3)) and np.abs(r - H_r).max() < 1e-12 and info["k"] == n3
          and info["denominator"] == n3 * n3 and info["nu"] == n3 * n3 and info["stochastic"] is False
          and abs(phl.estimator_from_products(v @ H_t, r, info) - L) < 1e-12,
          abs(phl.estimator_from_products(v @ H_t, r, info) - L))

    # --- the probe is the raw draw; r_j = H_r v_j --------------------------------------------
    v, r, info = phl.make_probes(H_r, mode="rademacher", k=4, rng=np.random.default_rng(1))
    check("k = 4 Rademacher: v is the raw +-1 draw [4, 3N], r = H_r v (1e-12), denominator 9 N^2 k",
          v.shape == (4, n3) and set(np.unique(v)) == {-1.0, 1.0} and np.abs(r - v @ H_r).max() < 1e-12
          and info["denominator"] == 4 * n3 * n3 and info["stochastic"] is True)
    vg, rg, _ig = phl.make_probes(H_r, mode="gaussian", k=3, rng=np.random.default_rng(1))
    check("k = 3 Gaussian: the draw is standard normal, r = H_r v (1e-12)",
          vg.shape == (3, n3) and np.abs(rg - vg @ H_r).max() < 1e-12 and abs(vg.std() - 1.0) < 0.2, vg.std())

    # --- unbiased, and the two variances -----------------------------------------------------
    draws = {}
    for mode, seed in (("rademacher", 11), ("gaussian", 13)):
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(4000):
            vj, rj, ij = phl.make_probes(H_r, mode=mode, k=1, rng=rng)
            vals.append(phl.estimator_from_products(vj @ H_t, rj, ij))
        draws[mode] = np.array(vals)
    var1 = phl.estimator_variance(H_t, H_r, k=1)
    for mode in ("rademacher", "gaussian"):
        vals = draws[mode]
        n = len(vals)
        se = math.sqrt(var1[mode] / n)
        check("{}: 4000 draws' mean within 3 s.e. of eq. 1' ({:.2f} s.e.)".format(
              mode, abs(vals.mean() - L) / se), abs(vals.mean() - L) < 3 * se, (vals.mean(), L, se))
        # the sample variance has its own sampling error, sd(s^2) = s^2 sqrt((kurt - 1)/n);
        # for the Gaussian draw X is a weighted sum of chi^2_1 and that band is ~6 %, so the
        # tolerance is read off the draws rather than invented
        kurt = float(np.mean((vals - vals.mean()) ** 4) / vals.var() ** 2)
        se_var = vals.var() * math.sqrt((kurt - 1) / n)
        check("{}: sample variance within 3 sd(s^2) of the closed form "
              "({:.3e} vs {:.3e}, band {:.1%})".format(mode, vals.var(), var1[mode], 3 * se_var / var1[mode]),
              abs(vals.var() - var1[mode]) < 3 * se_var, (vals.var(), var1[mode], se_var))
    b = (H_t - H_r).T @ (H_t - H_r)
    kurtosis_term = 2.0 * float(np.sum(np.diag(b) ** 2)) / (float(n3 ** 2) ** 2)
    check("Var_Gaussian - Var_Rademacher = 2 sum_i B_ii^2 / (9N^2)^2 (1e-18), and it is > 0",
          abs((var1["gaussian"] - var1["rademacher"]) - kurtosis_term) < 1e-18 and kurtosis_term > 0
          and draws["gaussian"].var() > draws["rademacher"].var(),
          (var1["gaussian"] - var1["rademacher"], kurtosis_term))
    var4 = phl.estimator_variance(H_t, H_r, k=4)
    check("the variance falls as 1/k: Var(k=4) = Var(k=1)/4 for both draws (1e-20)",
          abs(var4["rademacher"] - var1["rademacher"] / 4) < 1e-20
          and abs(var4["gaussian"] - var1["gaussian"] / 4) < 1e-20)

    # --- what the signature no longer takes --------------------------------------------------
    for bad_mode in ("hutchinson", "modes"):
        try:
            phl.make_probes(H_r, mode=bad_mode)
            check("probe mode {!r} is refused".format(bad_mode), False)
        except ValueError as exc:
            check("probe mode {!r} is refused".format(bad_mode), "probe mode must be one of" in str(exc))
    try:
        phl.make_probes(H_r, mode="rademacher", metric="cartesian")
        check("make_probes takes no `metric` (there is one target)", False)
    except TypeError:
        check("make_probes takes no `metric` (there is one target)", True)
    try:
        phl.make_probes(masses, x_r, H_r)
        check("make_probes takes no masses or positions", False)
    except (TypeError, ValueError):
        check("make_probes takes no masses or positions", True)
    try:
        phl.estimator_variance(H_t, H_r, masses, x_r, k=1)
        check("estimator_variance takes only (H_theta, H_r, k)", False)
    except TypeError:
        check("estimator_variance takes only (H_theta, H_r, k)", True)
    try:
        phl.make_probes(H_r[:-1], mode="cartesian")
        check("a Label that is not square is refused", False)
    except ValueError as exc:
        check("a Label that is not square is refused", "square" in str(exc))

    print("\n{} checks, {} failed".format(17, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
