from src.experiments.experiment import Experiment
from torch.utils.data import DataLoader
import torch
from typing import Literal
from src.train.history import SingleEpochHistory, MetricEntry, TrainHistory, EpochLosses

Optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}


class Trainer:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.dataset = experiment.get_dataset()
        self.config = experiment.config
        self.dataloader_train = self._get_dataloader(split="train")
        self.dataloader_val = self._get_dataloader(split="val")
        self.dataloader_test = self._get_dataloader(split="test")

        self.model = experiment.get_model()
        if self.config.optimize not in Optimizers:
            raise ValueError(
                f"Optimizer {self.config.optimizer} not implemented. "
                f"Choose from {Optimizers.keys()} or implement your own."
            )
        self.optimizer = Optimizers[self.config.optimizer](
            self.model.parameters(), lr=self.config
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )
        self.loss_fn = experiment.get_loss_function()

    def _get_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        return DataLoader(
            self.experiment.get_dataset(split),
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.experiment.get_collate_fn(),
        )

    def _log_intermediate(
        self, batch: int, n_batches: int, epoch_history: SingleEpochHistory
    ):
        loss = epoch_history.get_last().word_error_rate
        running = epoch_history.get_average().word_error_rate
        print(
            f"Batch {batch + 1}/{n_batches} loss: {loss} running: {running}\r",
            end="",
        )

    def _train_epoch(self):
        losses = SingleEpochHistory()
        self.model.train()
        for i, data in enumerate(self.dataloader_train):
            inputs, labels = data
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            losses.add_batch_metric(MetricEntry(loss.item()))
            if (
                i % self.config.log_every_n_batches
                == self.config.log_every_n_batches - 1
            ):
                self._log_intermediate(i, len(self.dataloader_train), losses)
        return losses

    def _evaluate_epoch(self, dataloader: DataLoader):
        losses = SingleEpochHistory()
        self.model.eval()

        for i, data in enumerate(dataloader):
            inputs, labels = data

            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

            losses.add_batch_metric(MetricEntry(loss.item()))
            if (
                i % self.config.log_every_n_batches
                == self.config.log_every_n_batches - 1
            ):
                self._log_intermediate(i, len(dataloader), losses)
        return losses

    def train(self):
        history: list[SingleEpochHistory] = []

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            train_losses = self._train_epoch()
            val_losses = self._evaluate_epoch(self.dataloader_val)
            self.scheduler.step()

            print(
                f"\n\n{'='*20}\nFinished Epoch {epoch + 1}/{self.config.epochs}"
                f"train WER: {train_losses.get_average().word_error_rate} "
                f"val WER: {val_losses.get_average().word_error_rate}"
            )
            history.append(EpochLosses(train_losses, val_losses))

        test_losses = self._evaluate_epoch(self.dataloader_test)
        print(f"\nTest WER: {test_losses.get_average().word_error_rate}")
        return self.model, TrainHistory(history, test_losses)
