"""Kolmogorov-Smirnov drift test wrapper."""

from scipy.stats import ks_2samp


def ks_test(reference, current):
    result = ks_2samp(reference, current, alternative="two-sided", mode="auto")
    return float(result.statistic), float(result.pvalue)
