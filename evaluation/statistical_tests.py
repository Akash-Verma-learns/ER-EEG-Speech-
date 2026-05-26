"""
Statistical comparison between model pairs using McNemar's test.

For each pair of models: build the 2×2 contingency table from their
per-sample predictions on the same held-out folds, then run McNemar's
exact or chi-squared test (p < 0.05 threshold).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from scipy.stats import chi2


def mcnemar_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
    exact: bool = False,
) -> Tuple[float, float, np.ndarray]:
    """
    McNemar's test comparing two classifiers on the same test set.

    Parameters
    ----------
    preds_a, preds_b : (n_samples,) integer predictions
    labels           : (n_samples,) ground-truth labels
    exact            : if True use exact binomial; otherwise chi-squared with continuity

    Returns
    -------
    statistic : McNemar statistic
    p_value   : p-value
    table     : 2×2 contingency table [[n00, n01], [n10, n11]]
                n01 = A correct & B wrong
                n10 = A wrong  & B correct
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    n00 = int(np.sum( correct_a & correct_b))   # both correct
    n01 = int(np.sum( correct_a & ~correct_b))  # A correct, B wrong
    n10 = int(np.sum(~correct_a & correct_b))   # A wrong, B correct
    n11 = int(np.sum(~correct_a & ~correct_b))  # both wrong

    table = np.array([[n00, n01], [n10, n11]])

    b, c = n01, n10  # conventional notation

    if exact:
        # Exact binomial: p = 2 * min(P(X <= min(b,c)), P(X >= max(b,c)))
        from scipy.stats import binom
        n = b + c
        if n == 0:
            return 0.0, 1.0, table
        p = 2.0 * binom.cdf(min(b, c), n, 0.5)
        p = min(p, 1.0)
        statistic = float(abs(b - c))
    else:
        # Chi-squared with continuity correction
        if b + c == 0:
            return 0.0, 1.0, table
        statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
        p = 1.0 - chi2.cdf(statistic, df=1)

    return float(statistic), float(p), table


def compare_all_pairs(
    results: Dict[str, Dict],
    alpha: float = 0.05,
    exact: bool = False,
) -> Dict[str, Dict]:
    """
    Run McNemar's test for every pair of models in `results`.

    Parameters
    ----------
    results : dict mapping model_name → {all_preds, all_labels, mean_acc, ...}
    alpha   : significance threshold

    Returns
    -------
    comparisons : dict keyed by "ModelA_vs_ModelB" with test statistics
    """
    model_names = list(results.keys())
    comparisons = {}

    for name_a, name_b in combinations(model_names, 2):
        res_a = results[name_a]
        res_b = results[name_b]

        preds_a = res_a["all_preds"]
        preds_b = res_b["all_preds"]
        labels  = res_a["all_labels"]  # must be the same fold structure

        stat, p_val, table = mcnemar_test(preds_a, preds_b, labels, exact=exact)

        key = f"{name_a}_vs_{name_b}"
        comparisons[key] = {
            "statistic": stat,
            "p_value": p_val,
            "significant": p_val < alpha,
            "contingency_table": table.tolist(),
            "acc_a": results[name_a]["mean_acc"],
            "acc_b": results[name_b]["mean_acc"],
            "delta_acc": results[name_b]["mean_acc"] - results[name_a]["mean_acc"],
        }

    return comparisons


def print_comparison_table(comparisons: Dict[str, Dict], alpha: float = 0.05):
    """Print a formatted summary table of all pairwise McNemar tests."""
    header = (
        f"{'Comparison':<30} {'Acc_A':>7} {'Acc_B':>7} "
        f"{'Δ Acc':>7} {'χ²/stat':>9} {'p-value':>9} {'sig?':>6}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for key, vals in comparisons.items():
        sig = "YES *" if vals["significant"] else "no"
        print(
            f"{key:<30} "
            f"{vals['acc_a']:>7.4f} "
            f"{vals['acc_b']:>7.4f} "
            f"{vals['delta_acc']:>+7.4f} "
            f"{vals['statistic']:>9.3f} "
            f"{vals['p_value']:>9.4f} "
            f"{sig:>6}"
        )

    print("=" * len(header))
    print(f"Significance threshold: p < {alpha}")


def compute_per_class_f1(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Return per-class F1 scores (useful for valence vs arousal breakdown)."""
    from sklearn.metrics import f1_score, classification_report
    n_classes = len(np.unique(labels))
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    f1s = f1_score(labels, preds, average=None, zero_division=0)
    return {name: float(f1) for name, f1 in zip(class_names, f1s)}


def summarise_results(
    results: Dict[str, Dict],
    class_names: Optional[List[str]] = None,
) -> None:
    """Print a clean summary table of all model results."""
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'Acc mean':>10} {'Acc std':>9} {'F1 mean':>9} {'F1 std':>8}")
    print("-" * 60)
    for name, res in results.items():
        print(
            f"{name:<20} "
            f"{res['mean_acc']:>10.4f} "
            f"±{res['std_acc']:>8.4f} "
            f"{res['mean_f1']:>9.4f} "
            f"±{res['std_f1']:>7.4f}"
        )
    print("=" * 60)

    if class_names:
        print("\nPer-class F1 (full held-out set):")
        for name, res in results.items():
            f1s = compute_per_class_f1(res["all_preds"], res["all_labels"], class_names)
            f1_str = "  ".join(f"{k}={v:.3f}" for k, v in f1s.items())
            print(f"  {name}: {f1_str}")
