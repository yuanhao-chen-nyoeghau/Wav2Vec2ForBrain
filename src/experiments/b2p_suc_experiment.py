from edit_distance import SequenceMatcher

from typing import Any, Literal, Optional, cast

from torch.optim.optimizer import Optimizer
from src.datasets.batch_types import PhonemeSampleBatch
import numpy as np
import torch
from src.util.phoneme_helper import PHONE_DEF_SIL
from src.args.yaml_config import YamlConfigModel
from src.experiments.b2p2t_experiment import B2P2TArgsModel, B2P2TExperiment
from src.model.b2p_suc import B2PSUC
from src.model.b2tmodel import B2TModel, ModelOutput
from src.model.b2p_suc import B2PSUCArgsModel
from src.train.evaluator import Evaluator
from src.train.history import DecodedPredictionBatch, MetricEntry, SingleEpochHistory


class B2PSUCEvaluator(Evaluator):
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
            prediction_batch if self.mode == "test" else None,
        )

    def evaluate(self) -> SingleEpochHistory:
        return self.history

    def _calc_phoneme_error_rate(
        self, batch: PhonemeSampleBatch, predictions: ModelOutput
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

        return total_edit_distance / total_seq_length, DecodedPredictionBatch(
            predicted, labels
        )


class B2PSUCExperimentArgsModel(B2PSUCArgsModel, B2P2TArgsModel):
    suc_for_ctc_checkpoint: Optional[str]


class B2PSUCExperiment(B2P2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

        if self.config.suc_for_ctc_checkpoint is not None:
            cast(B2PSUC, self.model.neural_decoder).suc_for_ctc.load_state_dict(
                torch.load(self.config.suc_for_ctc_checkpoint, map_location="cuda"),
                strict=True,
            )

    def get_name(self) -> str:
        return "b2p_suc"

    @staticmethod
    def get_args_model():
        return B2PSUCExperimentArgsModel

    def _create_neural_decoder(self) -> B2TModel:
        return B2PSUC(self.config, self._get_in_size_after_preprocessing())

    def create_evaluator(self, mode: Literal["train", "val", "test"]) -> Evaluator:
        return B2PSUCEvaluator(mode)

    def create_optimizer(self) -> Optimizer:
        optim_cls: Any = self._get_optimizer_cls()

        return optim_cls(
            cast(B2PSUC, self.model.neural_decoder).encoder.parameters(),
            lr=self.base_config.learning_rate,
            weight_decay=self.base_config.weight_decay,
            eps=self.base_config.optimizer_epsilon,
        )

    def process_test_results(self, test_results: SingleEpochHistory):
        pass
