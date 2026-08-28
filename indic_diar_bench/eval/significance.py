"""Statistical significance analysis, section 18: bootstrap CIs, paired tests, effect size."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapCI:
    mean: float
    lower: float
    upper: float
    confidence: float


def bootstrap_ci(values: list[float], confidence: float = 0.95, n_resamples: int = 2000, seed: int = 0) -> BootstrapCI:
    """section 18.1: bootstrap confidence interval over per-recording metric values."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return BootstrapCI(mean=0.0, lower=0.0, upper=0.0, confidence=confidence)

    rng = np.random.default_rng(seed)
    resample_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)
    ])
    alpha = 1 - confidence
    lower, upper = np.quantile(resample_means, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(mean=float(arr.mean()), lower=float(lower), upper=float(upper), confidence=confidence)


@dataclass
class PairedTestResult:
    statistic: float
    p_value: float
    test: str


def paired_test(baseline: list[float], proposed: list[float], test: str = "wilcoxon") -> PairedTestResult:
    """section 18.2: compare baseline vs proposed on the SAME recordings (paired samples)."""
    if len(baseline) != len(proposed):
        raise ValueError("baseline and proposed must have the same number of paired recordings")
    if len(baseline) < 2:
        return PairedTestResult(statistic=0.0, p_value=1.0, test=test)

    from scipy import stats as scipy_stats

    if test == "wilcoxon":
        try:
            stat, p = scipy_stats.wilcoxon(baseline, proposed)
        except ValueError:
            # all-zero differences (identical paired values) -- wilcoxon is undefined, report no effect
            return PairedTestResult(statistic=0.0, p_value=1.0, test=test)
    elif test == "ttest":
        stat, p = scipy_stats.ttest_rel(baseline, proposed)
    else:
        raise ValueError(f"Unknown test '{test}', expected 'wilcoxon' or 'ttest'")

    return PairedTestResult(statistic=float(stat), p_value=float(p), test=test)


def cohens_d_paired(baseline: list[float], proposed: list[float]) -> float:
    """section 18.3: effect size (Cohen's d for paired samples) -- magnitude of improvement,
    complementary to the significance test's p-value."""
    diffs = np.asarray(baseline, dtype=np.float64) - np.asarray(proposed, dtype=np.float64)
    if len(diffs) < 2 or diffs.std(ddof=1) == 0:
        return 0.0
    return float(diffs.mean() / diffs.std(ddof=1))
