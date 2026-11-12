"""Baadal command-line entry point.

    python run_pipeline.py            # full evaluation (~10-20 min on a laptop) + figures + results table
    python run_pipeline.py --quick    # one scenario, one budget, no noise run (~1 min)
    python run_pipeline.py --figures  # only rebuild figures/table from results/results.json
"""

import argparse
import json

from baadal.experiments import RESULTS_DIR, run_all
from baadal.report_assets import make_all


def main():
    ap = argparse.ArgumentParser(description="Baadal: forecast -> flood map -> budget-aware QAOA planning")
    ap.add_argument("--quick", action="store_true", help="smoke test: one scenario, one budget")
    ap.add_argument("--figures", action="store_true", help="only regenerate figures and tables")
    args = ap.parse_args()

    if args.quick:
        run_all(scenarios=("exceptional",), budgets=[220], with_noise=False, out_name="results_quick.json")
        return
    if not args.figures:
        run_all()
    results = json.loads((RESULTS_DIR / "results.json").read_text(encoding="utf-8"))
    make_all(results)


if __name__ == "__main__":
    main()
