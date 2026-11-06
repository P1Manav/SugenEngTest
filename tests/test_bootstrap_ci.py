"""Unit test for bootstrap confidence intervals in evaluate.py.
Validates bootstrap CI against a known synthetic binomial proportion.
"""

import sys
sys.path.insert(0, ".")
import numpy as np
from evaluate import bootstrap_ci

def test_bootstrap_known_proportion():
    # 60 trials with 33 successes (exact 55% Top-1 rate as on dev)
    n = 60
    k = 33
    synthetic_results = [{"slice": "A", "top1": 1, "top3": 1} for _ in range(k)] + \
                        [{"slice": "A", "top1": 0, "top3": 0} for _ in range(n - k)]

    ci = bootstrap_ci(synthetic_results, n_resamples=2000, alpha=0.05)
    t1_low, t1_high = ci["top1_ci"]

    print(f"Synthetic n={n}, k={k} (p={k/n:.3f}):")
    print(f"  Top-1 95% Bootstrap CI: [{t1_low*100:.1f}%, {t1_high*100:.1f}%]")

    # Check that CI is centered around 55% and does NOT erroneously have an 85% upper bound
    assert 0.40 <= t1_low <= 0.45, f"Expected lower bound ~41-45%, got {t1_low}"
    assert 0.65 <= t1_high <= 0.70, f"Expected upper bound ~65-70%, got {t1_high}"
    assert t1_high < 0.75, f"Upper bound {t1_high} must not be inflated to 85%!"
    print("PASS: Bootstrap CI correctly computes symmetrical bounds around sample mean.")

if __name__ == "__main__":
    test_bootstrap_known_proportion()
