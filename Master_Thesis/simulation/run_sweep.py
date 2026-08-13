"""
Grid-search sweep runner for the network simulation.

Launches ``main_network.py`` once per combination in ``PARAM_GRID``, each as an
independent subprocess. Parameters pass via the ``SIM_*`` environment variables
read by ``config.py``, so the grid extends without editing ``config.py`` itself.

Each run gets a tagged log directory under ``logs/`` (via ``SIM_RUN_ID``); the
full grid is recorded in ``manifest.json`` alongside the sweep, joining results
back to the parameter combination that produced them.

Usage
-----
    python run_sweep.py                  # run the full grid sequentially
    python run_sweep.py --workers 3      # run up to 3 combinations in parallel
    python run_sweep.py --dry-run        # print the grid without running anything

Parallelization
----------------
Each grid point is a separate OS process running its own Ollama calls, so
``--workers > 1`` parallelizes at the process level. No shared Python state
between runs. The practical limit is Ollama's own concurrency (GPU/CPU
throughput), not this script. Start with ``--workers 1`` to measure a single
run's wall-clock time before increasing it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# -- Grid definition ----------------------------------------------------------
# Each key maps to the SIM_* environment variable read by config.py.
# Edit these lists to change the sweep; values are cast to str() when set.
PARAM_GRID: dict[str, list] = {
    "SIM_SBM_P_INTER": [0.05, 0.1, 0.3],
    "SIM_RESPONDER_SELECTION_BETA": [0.0, 2.0],
    "SIM_OPINION_BETA": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "SIM_GRAPH_DYNAMIC": ["0", "1"],
}

# Pilot-run overrides applied to every combination (kept small for fast sweeps).
# Remove or adjust once the pipeline is validated.
COMMON_OVERRIDES: dict[str, str] = {
    "SIM_NUM_AGENTS": "4",
    "SIM_NETWORK_MAX_ROUNDS": "3",
}


def build_grid() -> list[dict[str, str]]:
    """Expand ``PARAM_GRID`` into one env-var dict per combination."""
    keys = list(PARAM_GRID.keys())
    combos = []
    for values in itertools.product(*PARAM_GRID.values()):
        combo = {key: str(value) for key, value in zip(keys, values)}
        combos.append(combo)
    return combos


def run_one(combo: dict[str, str], sweep_dir: Path, index: int) -> dict:
    """Run a single grid point in a subprocess and return its manifest entry."""
    run_id = f"sweep_{sweep_dir.name}_{index:03d}"
    env = os.environ.copy()
    env.update(COMMON_OVERRIDES)
    env.update(combo)
    env["SIM_RUN_ID"] = run_id

    start = time.time()
    result = subprocess.run(
        [sys.executable, "main_network.py"],
        env=env,
        capture_output=True,
        text=True,
    )
    duration = time.time() - start

    return {
        "run_id": run_id,
        "params": combo,
        "common_overrides": COMMON_OVERRIDES,
        "returncode": result.returncode,
        "duration_seconds": round(duration, 1),
        "log_dir": f"logs/run_{run_id}",
        "stderr_tail": result.stderr[-2000:] if result.returncode != 0 else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a parameter grid sweep over main_network.py")
    parser.add_argument("--workers", type=int, default=1, help="Number of grid points to run in parallel")
    parser.add_argument("--dry-run", action="store_true", help="Print the grid without running anything")
    args = parser.parse_args()

    combos = build_grid()
    print(f"Grid size: {len(combos)} combinations  │  workers={args.workers}")
    for i, combo in enumerate(combos):
        print(f"  [{i:03d}] {combo}")

    if args.dry_run:
        return

    sweep_dir = Path("logs") / "sweeps" / str(int(time.time()))
    sweep_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, combo, sweep_dir, i) for i, combo in enumerate(combos)]
        for future in futures:
            entry = future.result()
            status = "ok" if entry["returncode"] == 0 else f"FAILED (rc={entry['returncode']})"
            print(f"  {entry['run_id']}: {status}  ({entry['duration_seconds']}s)  -> {entry['log_dir']}")
            manifest.append(entry)

    manifest_path = sweep_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
