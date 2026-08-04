"""V gate confusion metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Confusion:
    """Binary confusion matrix counts and derived rates."""

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative

    def to_dict(self) -> dict[str, float | int]:
        """Serialize counts and rates."""
        precision_den = self.true_positive + self.false_positive
        recall_den = self.true_positive + self.false_negative
        neg_den = self.false_positive + self.true_negative
        pos_den = self.true_positive + self.false_negative
        return {
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "total": self.total,
            "false_positive_rate": self.false_positive / neg_den if neg_den else 0.0,
            "false_negative_rate": self.false_negative / pos_den if pos_den else 0.0,
            "precision": self.true_positive / precision_den if precision_den else 0.0,
            "recall": self.true_positive / recall_den if recall_den else 0.0,
        }


def confusion_from_decisions(current: list[bool], oracle: list[bool]) -> Confusion:
    """Build confusion matrix from current gate and oracle decisions."""
    if len(current) != len(oracle):
        raise ValueError("current and oracle decisions must have equal length")
    tp = tn = fp = fn = 0
    for cur, opt in zip(current, oracle):
        if cur and opt:
            tp += 1
        elif not cur and not opt:
            tn += 1
        elif cur and not opt:
            fp += 1
        else:
            fn += 1
    return Confusion(tp, tn, fp, fn)
