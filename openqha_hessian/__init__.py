"""Hessian learning: the loss side of fine-tuning MACE-OFF23 on reference E-F-H Labels.

PRODUCTION. What lives here decides the fine-tuned potential's numbers.

This is the `openqha-hessian` distribution (`import openqha_hessian`): the training
side of openQHA, moved out of `openqha/training/` (2026-09) so that its churn -- the
campaign's loss, probes, judge, training run -- no longer appears in openQHA's history.
The dependency direction is one way: this package imports
`openqha.{data,store,thermochem,potentials}` and the mace fork, and **openQHA never
imports this package**. The entry points stay openQHA's
(`workflows/hessian_learning/05_train.py`, `06_judge.py`).

The pattern is mace-md's (`source-code/mace-md-master`): everything that can be written
against mace's PUBLIC API lives in this package -- the Hessian-vector product (`hvp`),
the probes and the Hessian loss (`phl`, `phl_loss`), the judge (`judge`), the training
run and the smoke fit (`run`, `smoke_fit`) --
and the little that must touch mace's internals (a Hessian field on the batch, an
external-loss hook) lives as generic commits on the fork `BloomDlwlrma/mace`, branch
`openqha-hessian`, whose commit every training Record carries
(`openqha.potentials.engine.mace_fork_info`). Nothing in openQHA's `extensions`
subpackage is involved: that subpackage is for optional cross-checks no production
number depends on, and the fine-tuned potential is a production number.

Derivations and the numbers every test here is checked against:
`docs/tutorials/archive/T03_openQHA_Theory_Projected_Hessian_Loss.ipynb` in the openQHA
checkout, the long form of `.scratch/hessian-learning-set/design-phl-loss.md`.
"""


def package_identity():
    """The two Record fields in which a training or judge Record names this package.

    HL_PACKAGE_VERSION: the installed distribution's version (`openqha-hessian`), read
    from the distribution metadata so the project file stays the single place a version
    is bumped; "unknown" when it cannot be read. HL_PACKAGE_COMMIT: this checkout's
    commit, read best-effort with the same rules as the fork's commit
    (`openqha.potentials.engine.checkout_commit`) -- a wheel or a missing `.git` answers
    "unknown" and never blocks a run. Both are report-only strings; nothing parses them.
    """
    import importlib.metadata
    from openqha.potentials.engine import checkout_commit
    try:
        version = importlib.metadata.version("openqha-hessian")
    except Exception:                                     # noqa: BLE001 -- a missing distribution is not a refusal
        version = "unknown"
    commit = checkout_commit(__file__)[0]
    return dict(HL_PACKAGE_VERSION=version, HL_PACKAGE_COMMIT=commit)
