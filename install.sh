#!/usr/bin/env bash
# install.sh -- put the training stack into the ACTIVE environment: the mace fork and
# this package (openqha-hessian), both installed editable.
#
# Contract:
#   * Run it inside an already-created, activated environment (conda or venv); it acts
#     on that environment through `python -m pip` and nothing else.
#   * It never creates environments, never activates anything, and never installs or
#     touches openQHA itself (openQHA is a checkout already on the interpreter's path).
#   * The fork source is a local checkout by default -- the sibling `../mace` when no
#     argument is given, or the path argument -- so the default run never fetches the
#     fork from the network. `--url` is the explicit opt-in that fetches the fork from
#     its GitHub branch instead (the CI runs it that way, from a machine that carries
#     no checkout).
#   * Both artifacts are editable by design: `mace_fork_info()` reads the fork's commit
#     from the `.git` beside the imported package, and the training side refuses a fork
#     it cannot name. There is no wheel path and no non-editable variant here.
#   * No `--no-deps` (the mace-md pattern): the environment owns the dependencies; a
#     missing one is filled by the same pip runs.
#   * Re-running is safe: uninstall mace-torch -> install the fork -> install this
#     package -> verify, every time.
#
# Usage:
#   bash install.sh                  # fork from the sibling checkout `../mace` (offline)
#   bash install.sh /path/to/mace    # fork from a checkout at that path
#   bash install.sh --url            # explicit opt-in: fetch the fork from GitHub
#
# The fork source is validated BEFORE anything is uninstalled or installed: it must
# exist and be a git checkout whose root carries the mace package (`mace/__init__.py`)
# -- the layout `mace_fork_info()` reads. The no-argument form applies the same rules
# to `../mace`; when that is absent or invalid it fails, naming both remedies: a path
# argument, or the explicit `--url` opt-in.
#
# Exit status: non-zero when a step fails or any verification item fails.
set -euo pipefail

# `#egg=` names the requirement before pip clones; without it, older pips (observed
# on 24.0) fail with "Could not detect requirement name". Repo and branch are unchanged.
MACE_URL="git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian#egg=mace-torch"
MACE_VERSION="0.3.16+openqha"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fail() { echo "install.sh: ERROR: $*" >&2; exit 1; }

[ $# -le 1 ] || fail "usage: bash install.sh [MACE_CHECKOUT_PATH | --url]"
command -v python >/dev/null 2>&1 || \
    fail "no 'python' on PATH -- activate the target environment first (this script never creates one)"

# --- the fork source: the sibling checkout by default; --url is the explicit opt-in ----
# MACE_HINT carries the two remedies into the validation failures of the defaulted run.
MACE_SRC=""
MACE_HINT=""
if [ $# -eq 0 ]; then
    MACE_SRC="$ROOT/../mace"
    MACE_HINT=" -- pass a checkout path ('bash install.sh /path/to/mace'), or use the explicit network opt-in ('bash install.sh --url')"
elif [ "$1" = "--url" ]; then
    MACE_SRC="$MACE_URL"
else
    MACE_SRC="$1"
fi

if [ "$MACE_SRC" != "$MACE_URL" ]; then
    if [ ! -e "$MACE_SRC" ]; then
        if [ -n "$MACE_HINT" ]; then
            fail "no mace checkout found (looked beside this repository: $MACE_SRC)$MACE_HINT"
        fi
        fail "mace checkout not found: $MACE_SRC"
    fi
    [ -d "$MACE_SRC" ] || fail "mace checkout is not a directory: $MACE_SRC$MACE_HINT"
    [ -e "$MACE_SRC/.git" ] || fail "not a git checkout (no .git): $MACE_SRC$MACE_HINT"
    [ -f "$MACE_SRC/mace/__init__.py" ] || \
        fail "no mace package at the checkout root (want $MACE_SRC/mace/__init__.py): $MACE_SRC$MACE_HINT"
    MACE_SRC="$(cd "$MACE_SRC" && pwd)"
    echo "==> fork source: local checkout $MACE_SRC"
else
    echo "==> fork source: $MACE_URL  (explicit --url opt-in)"
fi

echo "==> environment: $(python -V 2>&1) at $(command -v python)"

echo "==> [1/4] uninstalling any mace-torch distribution"
python -m pip uninstall -y mace-torch || true

echo "==> [2/4] installing the mace fork editable"
python -m pip install -e "$MACE_SRC"

echo "==> [3/4] installing openqha-hessian editable"
python -m pip install -e "$ROOT"

echo "==> [4/4] verifying the end state"
python - "$MACE_VERSION" "$ROOT" <<'PY'
"""install.sh's end-state check: what training imports must pass every probe.

argv: expected mace version, this repository's root. Prints the fork identity, then
exits non-zero with every failure listed if any probe failed.
"""
import pathlib
import re
import subprocess
import sys

expected_version = sys.argv[1]
repo_root = pathlib.Path(sys.argv[2])
problems = []

try:
    import mace
except Exception as exc:                                      # noqa: BLE001
    print("FAIL: import mace failed: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
    sys.exit(1)

module_file = getattr(mace, "__file__", None)
checkout, commit, dirty = None, None, None

if not module_file:
    problems.append("import mace did not resolve to a file (a namespace package from the "
                    "current working directory ahead on sys.path?)")
else:
    module = pathlib.Path(module_file).resolve()
    checkout = module.parent.parent                     # the checkout root: .git sits here
    if not (checkout / ".git").exists():
        problems.append("no .git beside the mace package: import mace resolves to {} "
                        "instead of an editable checkout".format(module))
    else:
        def git(*args):
            return subprocess.run(["git", "-C", str(checkout), *args],
                                  capture_output=True, text=True)
        rev = git("rev-parse", "HEAD")
        commit = rev.stdout.strip()
        if rev.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
            problems.append("cannot read the fork commit: {}".format(rev.stderr.strip() or commit))
        state = git("status", "--porcelain", "--untracked-files=no")
        dirty = bool(state.stdout.strip())
        if state.returncode != 0:
            problems.append("git status failed: {}".format(state.stderr.strip()))
        elif dirty:
            problems.append("the checkout has modified tracked files:\n" + state.stdout.strip())

version = getattr(mace, "__version__", None)
if version != expected_version:
    problems.append("mace.__version__ is {!r}; expected {!r}".format(version, expected_version))

try:
    from openqha.potentials import engine
except Exception as exc:                                      # noqa: BLE001
    fork_info = "skipped (openQHA not importable: {}: {})".format(type(exc).__name__, exc)
else:
    info = engine.mace_fork_info()
    fork_info = "{}  dirty={}  path={}".format(
        info.get("mace_fork_commit"), info.get("mace_fork_dirty"), info.get("mace_fork_path"))
    if not re.fullmatch(r"[0-9a-f]{40}", str(info.get("mace_fork_commit") or "")):
        problems.append("mace_fork_info() answers {!r} instead of a full commit".format(
            info.get("mace_fork_commit")))
    if info.get("mace_fork_dirty"):
        problems.append("mace_fork_info() reports the checkout dirty")

try:
    import openqha_hessian
    package_file = getattr(openqha_hessian, "__file__", "?")
except Exception as exc:                                      # noqa: BLE001
    package_file = None
    problems.append("import openqha_hessian failed: {}: {}".format(type(exc).__name__, exc))

try:
    out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    package_commit = out.stdout.strip() if out.returncode == 0 else "unknown"
except Exception:                                             # noqa: BLE001
    package_commit = "unknown"

print("  mace            {}  {}".format(version, module_file))
print("  fork commit     {}{}".format(commit or "?", "  DIRTY" if dirty else ""))
print("  mace_fork_info  {}".format(fork_info))
print("  openqha_hessian {}  (package commit {})".format(package_file, package_commit))

if problems:
    print(file=sys.stderr)
    for problem in problems:
        print("FAIL: " + problem, file=sys.stderr)
    sys.exit(1)

print()
print("ok: the fork is installed editable and openqha-hessian imports")
PY

echo "==> done"
