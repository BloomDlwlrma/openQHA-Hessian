"""Shared helpers for the moved tests. Not a test itself.

The moved tests keep reading openQHA's `tests/data/...` fixtures in place (reused, not
copied), so they need the checkout `openqha` was imported from: a walk up from the test
file only finds a checkout while the test lives INSIDE one, and these tests no longer
do. `openqha_src` reads the import rather than re-discovering it -- whatever openqha the
environment resolved is the checkout whose fixtures are read, so a test cannot run one
checkout's code against another checkout's data.
"""
import os
from pathlib import Path


def openqha_src():
    """The openQHA checkout holding the `openqha` package and its `tests/data` fixtures.

    `OPENQHA_SRC` overrides, for an openqha that is importable some other way (a wheel,
    or a checkout that is not the imported one).
    """
    override = os.environ.get("OPENQHA_SRC")
    if override:
        return Path(override).resolve()
    import openqha
    return Path(openqha.__file__).resolve().parents[1]
