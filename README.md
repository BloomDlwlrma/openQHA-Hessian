# openqha-hessian

The training side of openQHA, as a package: the projected Hessian loss that fine-tunes
MACE-OFF23 on reference E-F-H labels (the Hessian-vector product, the probes, the loss).
It imports the openQHA checkout (`openqha.{data,store,thermochem,potentials}`) and the
mace fork; openQHA never imports it.

## Install

The working stack is three things: an openQHA checkout on the interpreter's path, the
mace fork `BloomDlwlrma/mace` (branch `openqha-hessian`), and this package. Inside an
already-activated environment, one script installs the last two, both editable:

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

Machines that only evaluate do not need the editable fork: a mace wheel is tolerated
outside training, with the fork commit recorded as `unknown`.

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
