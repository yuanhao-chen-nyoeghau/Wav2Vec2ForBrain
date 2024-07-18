import os
import torch
from torch.optim.optimizer import Optimizer
from src.experiments.experiment import Experiment
from src.datasets.timit_dataset import (
    TimitAudioDataset,
    TimitAudioDatasetArgsModel,
    TimitSampleBatch,
)
from src.model.w2v_suc_model import W2VSUCArgsModel, W2VSUCModel
from src.datasets.brain2text_w_phonemes import PHONE_DEF_SIL
from src.model.b2tmodel import ModelOutput
from src.args.base_args import BaseExperimentArgsModel
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, cast
from torch.utils.data import DataLoader
from src.train.evaluator import Evaluator
from src.train.history import DecodedPredictionBatch, MetricEntry, SingleEpochHistory
from sklearn.metrics import precision_score, recall_score, accuracy_score


class TimitW2VSUCEvaluator(Evaluator):
    def __init__(self, mode: Literal["train", "val", "test"]):
        super().__init__(mode)
        self.history = SingleEpochHistory()

    def _track_batch(self, predictions: ModelOutput, sample: TimitSampleBatch):
        # calculate accuracy, precision, recall
        predicted_ids = predictions.logits.argmax(dim=-1)
        predicted_class_ids_np = predicted_ids.cpu().numpy()
        assert sample.target != None, "Target can't be none"

        target_class_ids_np = sample.target.cpu().numpy()

        accuracy = accuracy_score(target_class_ids_np, predicted_class_ids_np)
        precision = precision_score(
            target_class_ids_np,
            predicted_class_ids_np,
            average="macro",
            zero_division=0,
        )
        recall = recall_score(
            target_class_ids_np,
            predicted_class_ids_np,
            average="macro",
            zero_division=0,
        )
        additional_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

        predictions.metrics.update(additional_metrics)

        assert (
            predictions.loss != None
        ), "Loss is None. Make sure to set loss in ModelOutput"

        decoded_batch = DecodedPredictionBatch(
            [PHONE_DEF_SIL[id] for id in predicted_class_ids_np],
            [PHONE_DEF_SIL[id] for id in target_class_ids_np],
        )
        self.history.add_batch_metric(
            MetricEntry(predictions.metrics, predictions.loss.cpu().item()),
            decoded_batch,
        )

    def evaluate(self) -> SingleEpochHistory:
        return self.history


class TimitW2VSUCExperimentArgsModel(
    BaseExperimentArgsModel, W2VSUCArgsModel, TimitAudioDatasetArgsModel
):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h", "facebook/wav2vec2-base-960h"
    ] = "facebook/wav2vec2-base-960h"
    unfreeze_strategy: Literal["suc"] = "suc"


class TimitW2VSUCExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = TimitW2VSUCExperimentArgsModel(**config)
        self.datasets: dict[str, TimitAudioDataset] = {
            "train": TimitAudioDataset(self.config, yamlConfig, split="train"),
            "val": TimitAudioDataset(self.config, yamlConfig, split="val"),
            "test": TimitAudioDataset(self.config, yamlConfig, split="test"),
        }
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "timit_w2v_suc"

    @staticmethod
    def get_args_model():
        return TimitW2VSUCExperimentArgsModel

    def _create_model(self):
        assert (
            self.config.loss_function == "cross_entropy",  # type: ignore
            "Only cross entropy loss is currently supported",
        )
        model = W2VSUCModel(self.config, self.datasets["train"].class_weights)
        return model

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "suc":
                return cast(W2VSUCModel, self.model).suc.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for wav2vec experiment"
            )

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)

    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        return self.datasets[split]

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        ds = self._create_dataset(split)
        return DataLoader(
            self._create_dataset(split),
            batch_size=self.base_config.batch_size,
            shuffle=True,
            collate_fn=ds.get_collate_fn(),
        )

    def get_vocab(self) -> list[str]:
        return PHONE_DEF_SIL

    def create_evaluator(self, mode: Literal["train", "val", "test"]):
        return TimitW2VSUCEvaluator(mode)

    def store_trained_model(self, trained_model: W2VSUCModel):
        torch.save(
            trained_model.suc.state_dict(),
            os.path.join(self.results_dir, "suc.pt"),
        )
