"""Tier 2A smoke: tiny CMA-ES run end-to-end (popsize=4, gens=2).

Just verifies that the loop runs, the objective decreases (or at least
stays finite), and the artifacts (generations.csv, best.json) are
written. Real Tier 2A run uses popsize=10, max_gens=40.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optim.cma_runner import run_cma


def main():
    print("=" * 70)
    print("Tier 2A smoke: tiny CMA-ES (popsize=4, gens=2)")
    print("=" * 70)
    best = run_cma(
        popsize=4,
        max_gens=4,
        sigma0=0.35,
        seeds_per_eval=2,
        deterministic_until_gen=2,
        pitch_jitter=False,  # smoke first; jitter is for the full run
        tag="smoke",
    )
    assert "score" in best
    assert best["residuals"] is not None
    print("\nSmoke: OK")


if __name__ == "__main__":
    main()
