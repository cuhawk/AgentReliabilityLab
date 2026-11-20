"""Binary judge calibration against paired gold human decisions."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from statistics import NormalDist

from arl.evaluation.metrics import wilson_interval


@dataclass(frozen=True)
class Calibration:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    kappa: float | None
    tpr: float | None
    tnr: float | None


@dataclass(frozen=True)
class CorrectedPassRate:
    observed: float
    corrected: float | None
    interval: tuple[float, float] | None


def calibrate(*, gold: Sequence[bool], judged: Sequence[bool]) -> Calibration:
    """Pair binary labels in sample order. Undefined statistics return None."""
    if len(gold) != len(judged):
        raise ValueError("gold and judged labels must have equal length")
    if any(type(label) is not bool for labels in (gold, judged) for label in labels):
        raise ValueError("calibration labels must be booleans")
    tp = tn = fp = fn = 0
    for actual, predicted in zip(gold, judged, strict=True):
        if actual and predicted:
            tp += 1
        elif not actual and not predicted:
            tn += 1
        elif predicted:
            fp += 1
        else:
            fn += 1

    n = len(gold)
    expected_agreements = (tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)
    denominator = n * n - expected_agreements
    kappa = ((tp + tn) * n - expected_agreements) / denominator if denominator else None
    return Calibration(
        tp, tn, fp, fn, kappa,
        tp / (tp + fn) if tp + fn else None,
        tn / (tn + fp) if tn + fp else None,
    )


def rogan_gladen(
    calibration: Calibration, *, judged_passes: int, total: int, confidence: float = 0.95,
) -> CorrectedPassRate:
    """Correct a separate batch's pass rate with the calibration confusion matrix.

    Project Bonferroni-adjusted Wilson intervals for the observed rate, TPR and
    TNR through the correction. Return the full range if the denominator could
    cross zero. Assumes the calibration rates transfer to the target batch.
    """
    if total <= 0 or not 0 <= judged_passes <= total:
        raise ValueError("expected 0 <= judged_passes <= total and total > 0")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    observed = judged_passes / total
    tpr, tnr = calibration.tpr, calibration.tnr
    if tpr is None or tnr is None:
        return CorrectedPassRate(observed, None, None)

    def correct(rate: float, sensitivity: float, specificity: float) -> float:
        return min(1.0, max(0.0, (rate + specificity - 1) / (sensitivity + specificity - 1)))

    identifiable = calibration.true_positive * calibration.true_negative != (
        calibration.false_positive * calibration.false_negative
    )
    corrected = correct(observed, tpr, tnr) if identifiable else None
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 6)
    rate_ci = wilson_interval(judged_passes, total, z)
    tpr_ci = wilson_interval(calibration.true_positive, calibration.true_positive + calibration.false_negative, z)
    tnr_ci = wilson_interval(calibration.true_negative, calibration.true_negative + calibration.false_positive, z)
    assert rate_ci is not None and tpr_ci is not None and tnr_ci is not None
    if tpr_ci[0] + tnr_ci[0] <= 1 <= tpr_ci[1] + tnr_ci[1]:
        return CorrectedPassRate(observed, corrected, (0.0, 1.0))
    corners = [correct(rate, sensitivity, specificity)
               for rate, sensitivity, specificity in product(rate_ci, tpr_ci, tnr_ci)]
    return CorrectedPassRate(observed, corrected, (min(corners), max(corners)))
