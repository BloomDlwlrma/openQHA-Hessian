# openqha-hessian

The training side of openQHA, as a package: the Projected Hessian Learning (PHL) loss that
fine-tunes MACE-OFF23 on reference E-F-H labels (the Hessian-vector product, the probes,
the loss). It imports the openQHA checkout (`openqha.{data,store,thermochem,potentials}`)
and the mace fork.

> **Important:** custom Hessian training in this repository does not work with the
> current unmodified upstream mace package. It needs the fork `BloomDlwlrma/mace`
> (branch `openqha-hessian`), installed editable: the upstream package cannot read the
> Hessian label, take a loss from outside, or keep the force graph at evaluation.

- openQHA -- https://github.com/BloomDlwlrma/openQHA -- the library this package trains
  against.
- mace -- https://github.com/BloomDlwlrma/mace, branch `openqha-hessian` -- the fork the
  fine-tuning runs on.

## How PHL enters MACE

The quantity PHL adds to the usual energy and force supervision is the mean squared
error per matrix element of the Cartesian Hessian, `L_H = ||H_theta - H_r||_F^2 /
(9 N^2)`: `H_theta` the model's second derivatives of the energy with respect to the
atomic positions, `H_r` the reference label at the same geometry. The label is used as
stored -- no mass weighting, no projection, no diagonalisation; the vibrational analysis
belongs to the judge.

Forming the full matrix would cost a differentiated second derivative per column --
`2 + 6N` forward-equivalents per batch. PHL compares Hessian-vector products on random
probes instead, `L^(k) = sum_j ||H_theta v_j - H_r v_j||^2 / (9 N^2 k)`: for zero-mean,
unit-covariance draws (`E[v] = 0`, `E[v v^T] = I`) this is an unbiased estimator of
`L_H` for every `k`. One product is one backward pass over the force graph
(`H v = -grad_x (F . v)`), so a step costs `2 + 2k` forward-equivalents for `k` probes.
The production draw is the standard normal (PHL's Algorithm 1); Rademacher probes have
the smaller variance; the deterministic `3N` unit probes make the estimator exact.
Training draws fresh probes every step; validation uses four standard-normal probes per
frame that the Dataset drew once and stored.

The model-side pieces live on the mace fork (a Hessian field on the batch, the
external-loss hook, the force graph at evaluation); everything writable against mace's
public API lives here, and the loss is reached from mace by name:

```text
--loss external --loss_module openqha_hessian.phl_loss:build
```

- `openqha_hessian.hvp` -- the Hessian-vector products from mace's force graph
  (`hvp_from_forces`), one backward pass per probe, no matrix formed.
- `openqha_hessian.phl` -- the probes and the reference matvec: `make_probes` (gaussian /
  rademacher / cartesian), `r_j = H_r v_j`, the estimator, and `loss_full`, the exact
  target.
- `openqha_hessian.phl_loss` -- the mace loss module (`build`): fresh probes every
  training step, the Dataset's fixed probes at evaluation; the energy and force terms
  are mace's own.
- `openqha_hessian.run` -- the fine-tune driver: builds mace's arguments, calls
  `mace.cli.run_train.run`, writes the Record; refuses a mace that is not the fork or a
  dirty checkout.
- `openqha_hessian.smoke_fit` -- the measuring fit: the `w_H` balance, the cost per
  probe setting, the exact-loss ceiling, the replay ratio.
- `openqha_hessian.judge` -- the ruler: the full Cartesian Hessian from `get_hessian`
  against the label, never the estimator.

The workflow around them is openQHA's `workflows/hessian_learning/`: `00_draw` and
`01_select` pick the molecules, `02_frames` generates the frames, `03_labels` attaches
the reference E-F-H labels, `04_dataset` assembles the Dataset, `05_train` fine-tunes
through `run`, and `06_judge` judges through `judge`.

> If you fine-tune with this package's Hessian loss, cite the PHL paper: Rodriguez,
> Smith, Matin, Lubbers, Barros & Mendoza-Cortes, *Projected Hessian Learning: Fast
> Curvature Supervision for Accurate Machine-Learning Interatomic Potentials*,
> arXiv:2603.04523 (key `rodriguez2026projectedhessianlearningfast`).

## Install

The working stack is three things: an openQHA checkout on the interpreter's path, the
mace fork `BloomDlwlrma/mace` (branch `openqha-hessian`), and this package. To enable
Hessian labels and the external loss, the fork must be installed editable; inside an
already-activated environment, one script does it together with this package:

```bash
bash install.sh                  # the fork from its GitHub branch (needs network)
bash install.sh /path/to/mace    # the fork from a local checkout (offline / Tianhe)
```

`install.sh` uninstalls any `mace-torch` wheel, editable-installs the fork and this
package, and verifies its own end state -- import location, version `0.3.16+openqha`,
tracked-clean checkout, `mace_fork_info()` full commit, `import openqha_hessian` -- and
exits non-zero if any of it fails. Re-running is safe.

Run it last -- or again after any re-run of openQHA's `install_dependency.sh`: that
installer reinstalls the fork from its requirement line (non-editable) and would replace
the editable install.

Editable is not an accident: `mace_fork_info()` reads the fork's commit from the `.git`
beside the imported package, and the training side refuses a fork it cannot name.

By hand, the same steps:

```bash
python -m pip uninstall -y mace-torch
python -m pip install -e "git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian#egg=mace-torch"
python -m pip install -e .
```

## Verify

```bash
python tests/run_tests.py            # unit group (seconds)
python tests/run_tests.py --all      # + integration (needs the MACE-OFF23 weights)
```

The runner prints, before anything runs, which openQHA checkout, package and mace this
interpreter loaded -- a failing test is unreadable without it. The fork identity alone,
the way training reads it:

```bash
python -c "from openqha.potentials.engine import mace_fork_info; print(mace_fork_info())"
```

## Citation

If you use openQHA or openQHA-Hessian in your work, please cite it:

```bibtex
@misc{openqha,
    title = {{openQHA}: conformational free energy for small organic molecules on a machine-learned potential},
    author = {Zhang, Shiwei},
    url = {https://github.com/BloomDlwlrma/openQHA},
    urldate = {2026-09-04},
    version = {0.1},
    year = {2026},
    month = sep,
    note = {Accessed: Sep 4, 2026},
}
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0
International License.
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
