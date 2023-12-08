from typing import NamedTuple
from dataclasses import dataclass


@dataclass
class MetricEntry:
    loss: float = 0

    def __iadd__(self, other: "MetricEntry"):
        self.loss += other.loss
        return self

    def __truediv__(self, other: float):
        return MetricEntry(self.loss / other)


class SingleEpochHistory:
    def __init__(self):
        self.metrics: list[MetricEntry] = []
        self._total_loss = MetricEntry(loss=0.0)
        self._total_loss_count = 0

    def add_batch_metric(self, loss: MetricEntry):
        self.metrics.append(loss)
        self._total_loss += loss
        self._total_loss_count += 1

    def get_average(self):
        return self._total_loss / self._total_loss_count

    def get_last(self):
        return self.metrics[-1]

    def to_dict(self):
        return {
            "history": [metric.__dict__ for metric in self.metrics],
            "average": self.get_average().__dict__,
        }


class EpochLosses(NamedTuple):
    train_losses: SingleEpochHistory
    val_losses: SingleEpochHistory

    def to_dict(self):
        return {
            "train": self.train_losses.to_dict(),
            "val": self.val_losses.to_dict(),
        }


class TrainHistory(NamedTuple):
    epochs: list[EpochLosses]
    test_losses: SingleEpochHistory

    def to_dict(self):
        return {
            "epochs": [epoch.to_dict() for epoch in self.epochs],
            "test": self.test_losses.to_dict(),
        }
