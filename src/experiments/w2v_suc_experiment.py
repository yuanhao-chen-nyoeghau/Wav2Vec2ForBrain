import os
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from src.datasets.audio_with_phonemes_seq import (
    AudioWPhonemesSeqDataset,
    AudioWPhonemesDatasetArgsModel,
)
from src.datasets.batch_types import PhonemeSampleBatch
from src.datasets.brain2text_w_phonemes import PHONE_DEF_SIL
from src.model.b2tmodel import ModelOutput
from src.model.w2v_suc_seq_model import W2VSUCSeqArgsModel, W2VSUCSeqModel
from src.args.base_args import BaseExperimentArgsModel
from src.model.audio_wav2vec_model import AudioWav2VecModel
from src.experiments.experiment import Experiment
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, cast
from datasets import load_dataset
from torch.utils.data import DataLoader
from src.train.evaluator import Evaluator
from src.train.history import DecodedPredictionBatch, MetricEntry, SingleEpochHistory
from edit_distance import SequenceMatcher


class W2VSUCEvaluator(Evaluator):
    def __init__(self, mode: Literal["train", "val", "test"]):
        super().__init__(mode)
        self.history = SingleEpochHistory()

    def _track_batch(self, predictions: ModelOutput, sample: PhonemeSampleBatch):
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
        self, batch: PhonemeSampleBatch, predictions: ModelOutput
    ):

        if batch.target == None or batch.target_lens == None:
            raise Exception("target or target_lens is None.")
        pred = predictions.logits
        total_edit_distance = 0
        total_seq_length = 0
        labels = []
        predicted = []
        for iterIdx in range(pred.shape[0]):
            decodedSeq = torch.argmax(
                torch.tensor(pred[iterIdx, :, :]),
                dim=-1,
            )  # [num_seq,]
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])
            trueSeq = np.array(
                batch.target[iterIdx][0 : batch.target_lens[iterIdx]].cpu().detach()
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

        return total_edit_distance / total_seq_length, DecodedPredictionBatch(
            [" ".join(p) for p in predicted], [" ".join(l) for l in labels]
        )


class W2VSUCExperimentArgsModel(
    BaseExperimentArgsModel, W2VSUCSeqArgsModel, AudioWPhonemesDatasetArgsModel
):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h", "facebook/wav2vec2-base-960h"
    ] = "facebook/wav2vec2-base-960h"
    unfreeze_strategy: Literal["suc"] = "suc"


class W2VSUCExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        base_dir = os.path.join(yamlConfig.cache_dir, "audio")
        cache_dir = os.path.join(base_dir, "cache")
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        self._hugg_dataset = load_dataset(
            "google/fleurs", name="en_us", cache_dir=cache_dir, data_dir=data_dir
        )
        self.config = W2VSUCExperimentArgsModel(**config)
        super().__init__(config, yamlConfig)
        self.model: AudioWav2VecModel = self.model

    def get_name(self) -> str:
        return "w2v_suc"

    @staticmethod
    def get_args_model():
        return W2VSUCExperimentArgsModel

    def _create_model(self):
        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        model = W2VSUCSeqModel(self.config)
        return model

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "suc":
                return cast(W2VSUCSeqModel, self.model).suc.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for wav2vec experiment"
            )

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)

    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        return AudioWPhonemesSeqDataset(self.config, self.yaml_config, split=split)

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
        return W2VSUCEvaluator(mode)
