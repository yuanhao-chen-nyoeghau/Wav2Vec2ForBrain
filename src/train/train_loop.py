from experiments.b2t_gru_transformer_experiment import B2tGruTrafoExperiment
from src.experiments.experiment import Experiment
from torch.utils.data import DataLoader
import torch
from typing import Literal, cast
from src.train.history import SingleEpochHistory, MetricEntry, TrainHistory, EpochLosses
import os
import wandb
from transformers.modeling_outputs import CausalLMOutput
from torcheval.metrics import WordErrorRate


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

    def _calculate_wer(
        self, predicted_logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        predicted_ids = predicted_logits.argmax(dim=-1).cpu().numpy()
        predicted_strings = self.experiment.tokenizer.batch_decode(
            predicted_ids, group_tokens=True
        )
        if self.config.experiment_type == "b2t_gru+trafo":
            label_strings = cast(
                B2tGruTrafoExperiment, self.experiment
            ).wav2vec_tokenizer.batch_decode(labels.cpu().numpy(), group_tokens=False)
        else:
            label_strings = self.experiment.tokenizer.batch_decode(
                labels.cpu().numpy(), group_tokens=False
            )

        # remove characters after EOS token
        def cut_after_eos_token(string: str):
            eos_token = "</s>"
            index_of_eos = string.find(eos_token)
            if index_of_eos != -1:
                return string[: (index_of_eos + len(eos_token))]
            else:
                return string

        predicted_strings = [
            cut_after_eos_token(string) for string in predicted_strings
        ]

        return (
            WordErrorRate()
            .update(input=predicted_strings, target=label_strings)
            .compute()
            .item()
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

            outputs.metrics["word_error_rate"] = self._calculate_wer(
                predicted_logits=outputs.logits, labels=labels
            )
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

        for i, data in enumerate(dataloader):
            inputs, labels = data

            with torch.no_grad():
                outputs = self.model.forward(inputs.cuda(), labels.cuda())

            outputs.metrics["word_error_rate"] = self._calculate_wer(
                predicted_logits=outputs.logits, labels=labels
            )
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

    def train(self):
        history: list[EpochLosses] = (
            self.experiment.checkpoint_history.epochs
            if not self.experiment.checkpoint_history is None
            else []
        )
        best_model_val_metric = float(
            "inf" if self.config.minimize_best_model_metric else "-inf"
        )
        best_model_path = os.path.join(self.yaml_config.cache_dir, "best_model.pt")

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
            history.append(EpochLosses(train_losses, val_losses))
            if self.config.return_best_model:
                curr_epoch_val_metric = (
                    val_losses.get_average().loss
                    if self.config.best_model_metric == "loss"
                    else val_losses.get_average().metrics[self.config.best_model_metric]
                )

                is_better = (
                    curr_epoch_val_metric < best_model_val_metric
                    if self.config.minimize_best_model_metric
                    else curr_epoch_val_metric > best_model_val_metric
                )
                if is_better:
                    best_model_val_metric = curr_epoch_val_metric
                    torch.save(self.model.state_dict(), best_model_path)
                    print("\n\nSaving model checkpoint\n")

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
