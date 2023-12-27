from src.experiments.experiment import Experiment
from torch.utils.data import DataLoader
import torch
from typing import Literal, cast
from src.train.history import SingleEpochHistory, MetricEntry, TrainHistory, EpochLosses
import os
import wandb
from transformers.modeling_outputs import CausalLMOutput


Schedulers = {"step": torch.optim.lr_scheduler.StepLR}


class Trainer:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.dataset = experiment._create_dataset()
        self.config = experiment.base_config
        self.yaml_config = experiment.yaml_config

        self.dataloader_train = experiment.dataloader_train
        self.dataloader_val = experiment.dataloader_val
        self.dataloader_test = experiment.dataloader_test

        self.model = experiment.model

        self.optimizer = experiment.create_optimizer()
        if self.config.scheduler not in Schedulers:
            raise ValueError(
                f"Scheduler {self.config.scheduler} not implemented. "
                f"Choose from {Schedulers.keys()} or implement your own."
            )
        self.scheduler = Schedulers[self.config.scheduler](
            self.optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma,
        )

    def _log_intermediate(
        self, batch: int, n_batches: int, epoch_history: SingleEpochHistory
    ):
        loss = epoch_history.get_last().loss
        running = epoch_history.get_average().loss
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
            with torch.enable_grad():
                # calculate gradient for whole model (but only optimize parts)
                outputs = self.model.forward(inputs.cuda(), labels.cuda())

            # Compute the loss and its gradients
            loss = cast(torch.Tensor, outputs.loss)
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
                outputs = self.model.forward(inputs.cuda(), labels.cuda())
                loss = cast(torch.Tensor, outputs.loss)

            losses.add_batch_metric(MetricEntry(loss.item()))
            if (
                i % self.config.log_every_n_batches
                == self.config.log_every_n_batches - 1
            ):
                self._log_intermediate(i, len(dataloader), losses)
        return losses

    def train(self):
        history: list[EpochLosses] = (
            self.experiment.checkpoint_history.epochs
            if not self.experiment.checkpoint_history is None
            else []
        )
        best_model_val_loss = float("inf")
        best_model_path = os.path.join(self.yaml_config.cache_dir, "best_model.pt")

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            train_losses = self._train_epoch()
            val_losses = self._evaluate_epoch(self.dataloader_val)
            self.scheduler.step()

            print(
                f"\n\n{'='*20}\nFinished Epoch {epoch + 1}/{self.config.epochs}"
                f"train {self.config.loss_function}-loss: {train_losses.get_average().loss} "
                f"val {self.config.loss_function}-loss: {val_losses.get_average().loss}"
            )
            history.append(EpochLosses(train_losses, val_losses))
            if self.config.return_best_model:
                curr_epoch_val_loss = val_losses.get_average().loss
                if curr_epoch_val_loss < best_model_val_loss:
                    best_model_val_loss = curr_epoch_val_loss
                    torch.save(self.model.state_dict(), best_model_path)

            wandb.log(
                {
                    f"train_{self.config.loss_function}_loss": train_losses.get_average().loss,
                    f"val_{self.config.loss_function}_loss": val_losses.get_average().loss,
                }
            )

        if self.config.return_best_model:
            self.model.load_state_dict(torch.load(best_model_path))
            os.remove(best_model_path)
            print("Loaded model with best validation loss of this experiment from disk")
        test_losses = self._evaluate_epoch(self.dataloader_test)
        print(
            f"\nTest loss ({self.config.loss_function}): {test_losses.get_average().loss}"
        )
        return self.model, TrainHistory(history, test_losses)
