"""Hessian learning: the loss side of fine-tuning MACE-OFF23 on reference E-F-H Labels.

PRODUCTION. What lives here decides the fine-tuned potential's numbers.

This is the `openqha-hessian` distribution (`import openqha_hessian`): the training
side of openQHA, leaving `openqha/training/` (2026-09; `hvp` moved first, its siblings
follow) so that its churn -- the campaign's loss, probes, judge, training run -- no
longer appears in openQHA's history.
The dependency direction is one way: this package imports
`openqha.{data,store,thermochem,potentials}` and the mace fork, and **openQHA never
imports this package**. The entry points stay openQHA's
(`workflows/hessian_learning/05_train.py`, `06_judge.py`).

The pattern is mace-md's (`source-code/mace-md-master`): everything that can be written
against mace's PUBLIC API lives in this package -- the Hessian-vector product (`hvp`),
the probes and the Hessian loss (`phl`, `phl_loss`), the judge (`judge`) --
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
