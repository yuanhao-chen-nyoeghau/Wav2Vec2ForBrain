from typing import NamedTuple
from dataclasses import dataclass


@dataclass
class MetricEntry:
    metrics: dict[str, float]
    loss: float = 0

    def __iadd__(self, other: "MetricEntry"):
        for key, value in other.metrics.items():
            if key in self.metrics and self.metrics[key] is not None:
                self.metrics[key] += value
            else:
                self.metrics[key] = value
        self.loss += other.loss
        return self

    def __truediv__(self, other: float):
        metrics_copy = dict(self.metrics)
        for key, value in metrics_copy.items():
            metrics_copy[key] /= other
        return MetricEntry(metrics_copy, self.loss / other)


class SingleEpochHistory:
    def __init__(self):
        self.metrics: list[MetricEntry] = []
        self._total_loss = MetricEntry({})
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

    @classmethod
    def from_json(cls, json_path: str):
        import json

        with open(json_path, "r") as f:
            data = json.load(f)

        epochs = data["epochs"]
        test_losses = data["test"]

        test_history = SingleEpochHistory()
        for metric in test_losses["history"]:
            test_history.add_batch_metric(MetricEntry(**metric))

        epoch_histories: list[EpochLosses] = []

        for epoch in epochs:
            train_epoch_history = SingleEpochHistory()
            for metric in epoch["train"]["history"]:
                train_epoch_history.add_batch_metric(MetricEntry(**metric))

            val_epoch_history = SingleEpochHistory()
            for metric in epoch["val"]["history"]:
                val_epoch_history.add_batch_metric(MetricEntry(**metric))

            epoch_history = EpochLosses(
                train_losses=train_epoch_history, val_losses=val_epoch_history
            )
            epoch_histories.append(epoch_history)

        return cls(
            epochs=epoch_histories,
            test_losses=test_history,
        )

    def plot(self, out_path: str):
        import matplotlib.pyplot as plt

        if len(self.epochs) == 0:
            return

        sample_epoch = self.epochs[0]
        train_metric_keys = sample_epoch.train_losses.metrics[0].metrics.keys()
        val_metric_keys = sample_epoch.val_losses.metrics[0].metrics.keys()
        # Creating a figure and subplots
        fig, ax = plt.subplots()
        for metric_key in train_metric_keys:
            # plot val and train loss history as subplots
            train_losses = [
                epoch.train_losses.get_average().metrics[metric_key]
                for epoch in self.epochs
            ]
            # Plotting train loss history
            ax.plot(
                train_losses, label=f"{metric_key} (train)", linestyle="-", marker="o"
            )

        for metric_key in val_metric_keys:
            val_losses = [
                epoch.val_losses.get_average().metrics[metric_key]
                for epoch in self.epochs
            ]
            # Plotting validation loss history on the same plot
            ax.plot(
                val_losses,
                label=f"{metric_key} (validation)",
                linestyle="-",
                marker="o",
            )

        # Adding labels and title
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Train and Validation Loss History")

        # Adding legend
        ax.legend()

        num_epochs = len(self.epochs)
        plt.xticks(list(range(num_epochs)), [str(i) for i in range(1, num_epochs + 1)])

        # Displaying the plot
        plt.show()
        plt.savefig(out_path)
