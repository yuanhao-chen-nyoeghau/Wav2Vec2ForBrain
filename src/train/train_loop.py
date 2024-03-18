from src.datasets.batch_types import SampleBatch
from src.model.b2tmodel import ModelOutput
from src.experiments.experiment import Experiment
from torch.utils.data import DataLoader
import torch
from typing import Literal, cast
from src.train.history import SingleEpochHistory, MetricEntry, TrainHistory, EpochLosses
import os
import wandb
from transformers.modeling_outputs import CausalLMOutput
from torcheval.metrics import WordErrorRate
import uuid
import numpy as np

Schedulers = {"step": torch.optim.lr_scheduler.StepLR}


class Trainer:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
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
            verbose=True,
        )

    def _log_intermediate(
        self, batch: int, n_batches: int, epoch_history: SingleEpochHistory
    ):
        loss = epoch_history.get_last().loss
        running = epoch_history.get_average().loss
        print(
            f"Batch {batch + 1}/{n_batches} loss: {loss:.2f} running: {running:.2f}\r",
            end="",
        )

    def _train_epoch(self):
        losses = SingleEpochHistory()
        self.model.train()

        for i, batch in enumerate(self.dataloader_train):
            batch = cast(SampleBatch, batch).cuda()

            self.optimizer.zero_grad()

            if self.config.whiteNoiseSD > 0:
                input, _ = batch
                noised_input = input + (
                    torch.randn(input.shape, device=input.device)
                    * self.config.whiteNoiseSD
                )
                batch._replace(input=noised_input)

            if self.config.constantOffsetSD > 0:
                input, _ = batch
                offset_input = input + (
                    torch.randn(
                        [input.shape[0], 1, input.shape[2]], device=input.device
                    )
                    * self.config.constantOffsetSD
                )
                batch._replace(input=offset_input)

            # Make predictions for this batch
            with torch.enable_grad():
                # calculate gradient for whole model (but only optimize parts)
                outputs = self.model.forward(batch)

            loss = cast(torch.Tensor, outputs.loss)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()
            additional_metrics = self.experiment.evaluate_batch(batch, outputs)
            outputs.metrics.update(additional_metrics)
            losses.add_batch_metric(MetricEntry(outputs.metrics, loss.cpu().item()))
            if (
                i % self.config.log_every_n_batches
                == self.config.log_every_n_batches - 1
            ):
                self._log_intermediate(i, len(self.dataloader_train), losses)
        return losses

    def _evaluate_epoch(self, dataloader: DataLoader):
        losses = SingleEpochHistory()
        self.model.eval()

        for i, batch in enumerate(dataloader):
            batch = cast(SampleBatch, batch).cuda()

            with torch.no_grad():
                outputs = self.model.forward(batch)

            additional_metrics = self.experiment.evaluate_batch(batch, outputs)
            outputs.metrics.update(additional_metrics)
            losses.add_batch_metric(
                MetricEntry(
                    outputs.metrics,
                    outputs.loss.item() if outputs.loss is not None else 0,
                )
            )
            if (
                i % self.config.log_every_n_batches
                == self.config.log_every_n_batches - 1
            ):
                self._log_intermediate(i, len(dataloader), losses)

        return losses

    def _get_wandb_metrics(self, epoch: SingleEpochHistory, prefix: str):
        def add_prefix_to_dict_keys(d: dict, prefix: str):
            return {f"{prefix}_{k}": v for k, v in d.items()}

        loss_avg = epoch.get_average()
        epoch_val_metrics = loss_avg.metrics

        wandb_metrics = {
            f"{prefix}_{self.config.loss_function}_loss": loss_avg.loss,
        }
        wandb_metrics.update(add_prefix_to_dict_keys(epoch_val_metrics, prefix))
        return wandb_metrics

    def _log_epoch_wandb(self, losses: EpochLosses):
        metrics = self._get_wandb_metrics(losses.val_losses, "val")
        metrics.update(self._get_wandb_metrics(losses.train_losses, "train"))
        wandb.log(metrics)

    def train(self):
        history: list[EpochLosses] = (
            self.experiment.checkpoint_history.epochs
            if not self.experiment.checkpoint_history is None
            else []
        )
        best_model_val_metric = float(
            "inf" if self.config.minimize_best_model_metric else "-inf"
        )
        best_model_path = os.path.join(
            self.yaml_config.cache_dir, str(uuid.uuid4()), "best_model.pt"
        )
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        def get_relevant_metric(epoch_hist: SingleEpochHistory):
            return (
                epoch_hist.get_average().loss
                if self.config.best_model_metric == "loss"
                else epoch_hist.get_average().metrics[self.config.best_model_metric]
            )

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            train_losses = self._train_epoch()
            val_losses = self._evaluate_epoch(self.dataloader_val)
            self.scheduler.step()

            print(
                f"\n\n{'='*20}\nFinished Epoch {epoch + 1}/{self.config.epochs} "
                f"train {self.config.loss_function}-loss: {train_losses.get_average().loss} "
                f"val {self.config.loss_function}-loss: {val_losses.get_average().loss}"
            )
            epoch_losses = EpochLosses(train_losses, val_losses)
            history.append(epoch_losses)
            self._log_epoch_wandb(epoch_losses)
            if self.config.return_best_model:
                curr_epoch_val_metric = get_relevant_metric(val_losses)

                is_better = (
                    curr_epoch_val_metric < best_model_val_metric
                    if self.config.minimize_best_model_metric
                    else curr_epoch_val_metric > best_model_val_metric
                )
                if is_better:
                    best_model_val_metric = curr_epoch_val_metric
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"\n\nSaving model checkpoint at {best_model_path}\n")

            if (
                self.config.early_stopping_patience is not None
                and len(history) >= self.config.early_stopping_patience
            ):
                relevant_metric_history = [
                    get_relevant_metric(epoch_loss.val_losses) for epoch_loss in history
                ][-self.config.early_stopping_patience :]

                best_index = (
                    np.argmin(relevant_metric_history)
                    if self.config.minimize_best_model_metric
                    else np.argmax(relevant_metric_history)
                )
                if best_index == 0:
                    print(
                        f"\nEarly stopping after {epoch} epochs ({self.config.early_stopping_patience} epochs without improvement in validation {self.config.best_model_metric} metrics)"
                    )
                    break

        if self.config.return_best_model:
            self.model.load_state_dict(torch.load(best_model_path))
            os.remove(best_model_path)
            os.rmdir(os.path.dirname(best_model_path))
            print("Loaded model with best validation loss of this experiment from disk")
        test_losses = self._evaluate_epoch(self.dataloader_test)
        wandb.log(self._get_wandb_metrics(test_losses, "test"))
        print(
            f"\nTest loss ({self.config.loss_function}): {test_losses.get_average().loss}"
        )
        return self.model, TrainHistory(history, test_losses)
