"""
run_pipeline.py
---------------
One-click runner: executes all analysis steps in order.

Usage:
    python src/run_pipeline.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("Initial readout (ATE + CI)", "src/initial_readout.py"),
    ("Uplift model (T-learner + deciles)", "src/uplift_baseline.py"),
    ("Segment uplift (BH correction)", "src/segment_uplift.py"),
    ("Policy simulation (breakeven)", "src/policy_simulation.py"),
]


def run(label: str, script: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script)],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    print("Advertising Uplift — Full Pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    for label, script in STEPS:
        run(label, script)
    print("\n[DONE] All steps completed successfully.")
    print("Outputs:")
    print("  reports/initial_readout.md")
    print("  reports/uplift_model.md")
    print("  reports/segment_uplift.md")
    print("  reports/policy_simulation.md")
    print("  data/processed/uplift_deciles.csv")
    print("  data/processed/uplift_scores.csv")
    print("  data/processed/segment_uplift.csv")
    print("  data/processed/policy_simulation.csv")


if __name__ == "__main__":
    main()
