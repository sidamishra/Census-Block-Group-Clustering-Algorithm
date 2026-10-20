"""Run the whole BG feature pipeline: feature modules -> standardize.

    python run_pipeline.py                      # build only missing interim files, then standardize
    python run_pipeline.py --force              # rebuild every interim file
    python run_pipeline.py --force crime        # rebuild just the named module(s)
    python run_pipeline.py --footprint-states 10 24   # footprint states if (re)built (default DE)
    python run_pipeline.py --skip-standardize

Each module is run as `python features/<name>.py` in a subprocess (its own __main__ does
build() + save_interim(), plus any reporting), so memory is released between modules.
building_footprints is optional enrichment and is NOT merged by standardize.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INTERIM = ROOT / "data" / "interim"
# order matters only for shared raw caches (geography's TIGER zips feed building_footprints)
MODULES = ["geography", "acs_demographics", "education", "crime", "occupation",
           "building_footprints"]


def run_module(name: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / "features" / f"{name}.py"), *(extra_args or [])]
    print(f"\n=== building {name}: {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", nargs="*", metavar="MODULE", default=None,
                    help="rebuild interim files (all if no module names given)")
    ap.add_argument("--footprint-states", nargs="*", default=None,
                    help="state FIPS for building_footprints (or 'all'); default Delaware")
    ap.add_argument("--skip-standardize", action="store_true")
    args = ap.parse_args(argv)

    force_all = args.force is not None and len(args.force) == 0
    forced = set(args.force or [])
    unknown = forced - set(MODULES)
    if unknown:
        ap.error(f"unknown module(s): {sorted(unknown)}; choose from {MODULES}")

    for name in MODULES:
        out = INTERIM / f"{name}.parquet"
        if out.exists() and not (force_all or name in forced):
            print(f"skip {name}: {out.relative_to(ROOT)} exists (use --force {name} to rebuild)")
            continue
        extra = args.footprint_states if name == "building_footprints" else None
        run_module(name, extra)

    if not args.skip_standardize:
        print("\n=== standardize", flush=True)
        import standardize
        standardize.main()


if __name__ == "__main__":
    main()
