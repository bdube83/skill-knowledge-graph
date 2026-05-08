"""Percentile bootstrap confidence intervals for the SKG evaluation.

Paper Section 7.5 specifies bootstrap resampling with n=1000 iterations
to produce 95% confidence intervals on continuous metrics. This module
implements that procedure as a small standalone helper.
"""

from __future__ import annotations

import random
import statistics


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    iterations: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute the percentile bootstrap mean and confidence interval.

    The function resamples the input list with replacement for the given
    number of iterations. Each resample yields a mean. The CI bounds are
    the lower and upper percentiles that bracket the requested confidence
    level.

    Returns (mean, ci_lower, ci_upper). The mean is the sample mean of
    the original input. The bounds come from the resampled distribution.

    Edge cases. An empty input returns (0.0, 0.0, 0.0). A single value
    returns that value as the mean and both bounds.
    """
    if not values:
        return 0.0, 0.0, 0.0
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must be in (0, 1); got {confidence}",
        )
    if iterations <= 0:
        raise ValueError(
            f"iterations must be positive; got {iterations}",
        )

    sample_mean = statistics.fmean(values)
    if len(values) == 1:
        return sample_mean, sample_mean, sample_mean

    rng = random.Random(seed)
    n = len(values)
    resample_means: list[float] = []
    for _ in range(iterations):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        resample_means.append(statistics.fmean(resample))

    resample_means.sort()
    alpha = 1.0 - confidence
    lower_idx = int(round((alpha / 2.0) * iterations))
    upper_idx = int(round((1.0 - alpha / 2.0) * iterations)) - 1
    lower_idx = max(0, min(lower_idx, iterations - 1))
    upper_idx = max(0, min(upper_idx, iterations - 1))

    return sample_mean, resample_means[lower_idx], resample_means[upper_idx]
