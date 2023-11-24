from typing import NamedTuple


class MetricEntry(NamedTuple):
    word_error_rate: float

    def __iadd__(self, other):
        self.word_error_rate += other.word_error_rate

    def __truediv__(self, other: float):
        return MetricEntry(self.word_error_rate / other)


class SingleEpochHistory:
    def __init__(self):
        self.metrics: list[MetricEntry] = []
        self._total_loss = MetricEntry(0.0)
        self._total_loss_count = 0

    def add_batch_metric(self, loss: MetricEntry):
        self.metrics.append(loss)
        self._total_loss += loss
        self._total_loss_count += 1

    def get_average(self):
        return self._total_loss / self._total_loss_count

    def get_last(self):
        return self.metrics[-1]


class EpochLosses(NamedTuple):
    train_losses: SingleEpochHistory
    val_losses: SingleEpochHistory


class TrainHistory(NamedTuple):
    epochs: list[EpochLosses]
    test_losses: SingleEpochHistory
