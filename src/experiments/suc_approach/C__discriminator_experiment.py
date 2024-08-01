import os
import torch
from torch.optim.optimizer import Optimizer
from src.datasets.discriminator_dataset import (
    DiscriminatorDataset,
    DiscriminatorDatasetArgsModel,
)
from src.model.discriminator_model import (
    DiscriminatorModel,
    DiscriminatorModelArgsModel,
)
from src.experiments.experiment import Experiment
from src.datasets.timit_dataset import (
    TimitSampleBatch,
)
from src.util.phoneme_helper import PHONE_DEF_SIL
from src.model.b2tmodel import ModelOutput
from src.args.base_args import BaseExperimentArgsModel
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal
from torch.utils.data import DataLoader
from src.train.evaluator import Evaluator
from src.train.history import MetricEntry, SingleEpochHistory
from sklearn.metrics import precision_score, recall_score, accuracy_score


class DiscriminatorEvaluator(Evaluator):
    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        super().__init__(mode, track_non_test_predictions)
        self.history = SingleEpochHistory()

    def _track_batch(self, predictions: ModelOutput, sample: TimitSampleBatch):
        # calculate accuracy, precision, recall
        predicted_classes = predictions.logits.round()
        predicted_classes_np = predicted_classes.detach().cpu().numpy()
        assert sample.target != None, "Target can't be none"

        target_classes_np = sample.target.detach().cpu().numpy()

        accuracy = accuracy_score(target_classes_np, predicted_classes_np)
        precision = precision_score(
            target_classes_np,
            predicted_classes_np,
            average="macro",
            zero_division=0,
        )
        recall = recall_score(
            target_classes_np,
            predicted_classes_np,
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

        self.history.add_batch_metric(
            MetricEntry(predictions.metrics, predictions.loss.cpu().item()),
        )

    def evaluate(self) -> SingleEpochHistory:
        return self.history


class DiscriminatorExperimentArgsModel(
    BaseExperimentArgsModel, DiscriminatorDatasetArgsModel, DiscriminatorModelArgsModel
):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints

    pass


class DiscriminatorExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = DiscriminatorExperimentArgsModel(**config)

        brain_feat_extr = DiscriminatorDataset.brain_feature_extractor_from_config(
            self.config, self.config.brain_encoder_path
        )
        w2v_feat_extr = DiscriminatorDataset.w2v_feature_extractor()
        self.datasets: dict[str, DiscriminatorDataset] = {
            "train": DiscriminatorDataset(
                self.config, yamlConfig, "train", w2v_feat_extr, brain_feat_extr
            ),
            "val": DiscriminatorDataset(
                self.config, yamlConfig, "val", w2v_feat_extr, brain_feat_extr
            ),
            "test": DiscriminatorDataset(
                self.config, yamlConfig, "test", w2v_feat_extr, brain_feat_extr
            ),
        }
        del w2v_feat_extr
        del brain_feat_extr

        import gc  # garbage collect library

        gc.collect()
        torch.cuda.empty_cache()
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "discriminator"

    @staticmethod
    def get_args_model():
        return DiscriminatorExperimentArgsModel

    def _create_model(self):
        assert (
            self.config.loss_function == "bce",  # type: ignore
            "Only bce loss is currently supported",
        )
        model = DiscriminatorModel(
            self.config,
            self.datasets["train"].n_w2v_samples
            / self.datasets["train"].n_brain_samples,
        )
        return model

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            return self.model.parameters()

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

    def create_evaluator(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        return DiscriminatorEvaluator(mode, track_non_test_predictions)

    def store_trained_model(self, trained_model: DiscriminatorModel):
        torch.save(
            trained_model.state_dict(),
            os.path.join(self.results_dir, "discriminator.pt"),
        )
