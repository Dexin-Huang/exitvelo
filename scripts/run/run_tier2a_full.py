"""Full Tier 2A: CMA-ES on the 3 active swing residuals.

Configuration per SPEC.md §3 (revised):
  popsize:          10
  max_gens:         40
  sigma0:           0.20  (Codex review: tighter than 0.35 — smoke found
                          contact in gen 1 with sigma already at 0.21)
  jitter:           OFF   (3-d residuals can't compensate for ball-
                          arrival variance under jitter; deferred to
                          Tier 2B PPO with closed-loop policy)
  seeds_per_eval:   1     (deterministic; no jitter -> single seed)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optim.cma_runner import run_cma


def main():
    print("=" * 70)
    print("Tier 2A FULL: CMA-ES on 3 named swing residuals")
    print("=" * 70)
    best = run_cma(
        popsize=10,
        max_gens=40,
        sigma0=0.20,
        seeds_per_eval=1,
        deterministic_until_gen=40,  # all gens deterministic (no jitter)
        pitch_jitter=False,
        tag="full_v1",
    )
    print("\nDone.")
    return best


if __name__ == "__main__":
    main()
