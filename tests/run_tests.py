"""Run the moved tests. Each test is a program that exits non-zero when it fails.

The twin of openQHA's `tests/run_tests.py` -- same conventions: one subprocess per file,
groups `unit`/`integration`, `--group`/`--all`, unit by default -- with one addition at
the top, the provenance header. These tests read three independently versioned things:
the openQHA checkout (fixtures, and half of every engine test's imports), this package,
and the mace fork. A failing test is unreadable without knowing which of the three was
loaded, and there is exactly one way to know: print it before anything runs.

Layout:

    unit/         pure functions and file-level checks; seconds, no engine
    integration/  needs an engine (MACE-OFF23); slow

Run::  python tests/run_tests.py
       python tests/run_tests.py --all
       python tests/run_tests.py --group integration
"""
import argparse
import importlib
import importlib.metadata
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
GROUPS = ("unit", "integration")


def provenance():
    """(name, value) rows: the interpreter, the two packages, the mace distribution."""
    rows = [("python", "{}  {}".format(sys.version.split()[0], sys.executable))]
    for name in ("openqha", "openqha_hessian"):
        try:
            module = importlib.import_module(name)
            rows.append((name, str(getattr(module, "__file__", "?"))))
        except Exception as exc:                       # noqa: BLE001 -- the header is the diagnosis
            rows.append((name, "NOT IMPORTABLE: {}: {}".format(type(exc).__name__, exc)))
    try:
        mace = importlib.metadata.version("mace-torch")
    except Exception as exc:                           # noqa: BLE001
        mace = "unknown: {}: {}".format(type(exc).__name__, exc)
    rows.append(("mace-torch", mace))
    return rows


def discover(groups):
    for g in groups:
        d = TESTS / g
        if not d.is_dir():
            continue
        for p in sorted(d.glob("t_*.py")):
            yield g, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run every group, not just unit")
    ap.add_argument("--group", choices=GROUPS, help="run one group")
    ap.add_argument("--verbose", action="store_true", help="show each test's output")
    args = ap.parse_args()

    rows = provenance()
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print("  {:<{w}s} {}".format(name, value, w=width))
    print()

    groups = (args.group,) if args.group else (GROUPS if args.all else ("unit",))
    tests = list(discover(groups))
    if not tests:
        print("no tests found in {}".format(", ".join(groups)))
        return 1

    print("=" * 92)
    print("running {} test(s) from {}".format(len(tests), ", ".join(groups)))
    print("=" * 92)
    failed = []
    for g, p in tests:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(p)], cwd=str(ROOT),
                           capture_output=not args.verbose, text=True)
        dt = time.time() - t0
        mark = "pass" if r.returncode == 0 else "FAIL"
        print("  {:<4s} {:<10s} {:<44s} {:6.2f}s".format(mark, g, p.name, dt))
        if r.returncode != 0:
            failed.append((p, r))

    print()
    if failed:
        for p, r in failed:
            print("-" * 92)
            print("FAILED: {}".format(p.relative_to(ROOT)))
            print("-" * 92)
            if not args.verbose:
                tail = (r.stdout or "").splitlines()[-24:]
                print("\n".join(tail))
                if r.stderr:
                    print(r.stderr.strip()[-1500:])
        print("\n{} of {} test(s) failed".format(len(failed), len(tests)))
        return 1
    print("all {} test(s) passed".format(len(tests)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
