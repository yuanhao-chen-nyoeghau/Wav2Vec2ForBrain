import os
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from src.experiments.w2v_suc_experiment import W2VSUCExperiment
from src.datasets.timit_ctc_dataset import TimitAudioSeqDataset, TimitSeqSampleBatch
from src.datasets.audio_with_phonemes_seq import (
    AudioWPhonemesDatasetArgsModel,
)
from src.util.phoneme_helper import PHONE_DEF_SIL
from src.model.b2tmodel import ModelOutput
from src.model.w2v_suc_ctc_model import W2VSUC_CTCArgsModel, W2VSUCForCtcModel
from src.args.base_args import BaseExperimentArgsModel
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, cast
from torch.utils.data import DataLoader
from src.train.evaluator import Evaluator
from src.train.history import DecodedPredictionBatch, MetricEntry, SingleEpochHistory
from edit_distance import SequenceMatcher


class EnhancedDecodedBatch(DecodedPredictionBatch):
    original_targets: list[str]


class TimitSeqW2VSUCEvaluator(Evaluator):
    def __init__(self, mode: Literal["train", "val", "test"]):
        super().__init__(mode)
        self.history = SingleEpochHistory()

    def _track_batch(self, predictions: ModelOutput, sample: TimitSeqSampleBatch):
        phoneme_error_rate, prediction_batch = self._calc_phoneme_error_rate(
            sample, predictions
        )
        additional_metrics = {"phoneme_error_rate": phoneme_error_rate}
        predictions.metrics.update(additional_metrics)

        assert (
            predictions.loss != None
        ), "Loss is None. Make sure to set loss in ModelOutput"
        self.history.add_batch_metric(
            MetricEntry(predictions.metrics, predictions.loss.cpu().item()),
            prediction_batch,
        )

    def evaluate(self) -> SingleEpochHistory:
        return self.history

    def _calc_phoneme_error_rate(
        self, batch: TimitSeqSampleBatch, predictions: ModelOutput
    ):
        pred = predictions.logits
        total_edit_distance = 0
        total_seq_length = 0
        labels = []
        predicted = []
        for iterIdx in range(pred.shape[0]):
            if batch.target is None:
                continue
            decodedSeq = torch.argmax(
                torch.tensor(pred[iterIdx, :, :]),
                dim=-1,
            )  # [num_seq,]
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])
            trueSeq = np.array(
                [idx.item() for idx in batch.target[iterIdx].cpu() if idx >= 0]
            )
            # TODO: is this correct?
            labels.append([PHONE_DEF_SIL[i - 1] for i in trueSeq])
            predicted.append([PHONE_DEF_SIL[i - 1] for i in decodedSeq])
            matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
            dist = matcher.distance()
            if dist == None:
                print(
                    "[evaluate batch]: distance from sequence matcher is None, skipping."
                )
                continue
            total_edit_distance += dist
            total_seq_length += len(trueSeq)

        decoded = EnhancedDecodedBatch(predicted, labels)
        decoded.original_targets = batch.transcripts

        return total_edit_distance / total_seq_length, decoded


class W2VSUCCtcExperimentArgsModel(
    BaseExperimentArgsModel, W2VSUC_CTCArgsModel, AudioWPhonemesDatasetArgsModel
):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h", "facebook/wav2vec2-base-960h"
    ] = "facebook/wav2vec2-base-960h"
    unfreeze_strategy: Literal["suc"] = "suc"
    suc_checkpoint: str


class TimitW2VSUC_CTCExperiment(W2VSUCExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        super().__init__(config, yamlConfig)
        self.config = self.get_args_model()(**config)

    def get_name(self) -> str:
        return "timit_w2v_suc_ctc"

    @staticmethod
    def get_args_model():
        return W2VSUCCtcExperimentArgsModel

    def _create_model(self):
        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is supported",
        )
        model = W2VSUCForCtcModel(self.config)
        model.suc_for_ctc.suc.load_state_dict(
            torch.load(self.config.suc_checkpoint, map_location="cuda"),
            strict=True,
        )
        return model

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "suc":
                return cast(
                    W2VSUCForCtcModel, self.model
                ).suc_for_ctc.ctc_head.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for wav2vec experiment"
            )

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)

    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        return TimitAudioSeqDataset(self.config, self.yaml_config, split=split)

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        ds = self._create_dataset(split)
        return DataLoader(
            self._create_dataset(split),
            batch_size=self.base_config.batch_size,
            shuffle=True,
            collate_fn=ds.get_collate_fn(),
        )

    def get_vocab(self) -> list[str]:
        return ["BLANK"] + PHONE_DEF_SIL

    def create_evaluator(self, mode: Literal["train", "val", "test"]):
        return TimitSeqW2VSUCEvaluator(mode)

    def store_trained_model(self, trained_model: W2VSUCForCtcModel):
        torch.save(
            trained_model.suc_for_ctc.state_dict(),
            os.path.join(self.results_dir, "suc_for_ctc.pt"),
        )
