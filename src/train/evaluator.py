from math import nan
from typing import Literal
from src.datasets.batch_types import SampleBatch
from src.model.b2tmodel import ModelOutput
from src.train.history import DecodedPredictionBatch, SingleEpochHistory
from abc import ABC, abstractmethod
from abc import abstractmethod
from src.datasets.batch_types import SampleBatch
from src.model.b2tmodel import ModelOutput
from src.train.history import MetricEntry, SingleEpochHistory
from torcheval.metrics import WordErrorRate
from transformers import PreTrainedTokenizer


class Evaluator(ABC):
    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        self.running_loss = 0.0
        self.n_losses = 0
        self.latest_loss = nan
        self.mode = mode
        self.track_non_test_predictions = track_non_test_predictions

    def track_batch(self, predictions: ModelOutput, sample: SampleBatch):
        assert predictions.loss is not None
        self.running_loss += predictions.loss.item()
        self.n_losses += 1
        self.latest_loss = predictions.loss.item()
        self._track_batch(predictions, sample)

    def get_running_loss(self):
        return self.running_loss / self.n_losses

    def get_latest_loss(self):
        return self.latest_loss

    @abstractmethod
    def _track_batch(self, predictions: ModelOutput, sample: SampleBatch):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self) -> SingleEpochHistory:
        raise NotImplementedError()

    def clean_up(self):
        pass


class DefaultEvaluator(Evaluator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        super().__init__(mode, track_non_test_predictions)
        self.history = SingleEpochHistory()
        self.tokenizer = tokenizer

    def _track_batch(self, predictions: ModelOutput, sample: SampleBatch):
        predicted_strings, label_strings = self.decode_predictions(predictions, sample)

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
        if label_strings is not None:
            additional_metrics = {
                "word_error_rate": WordErrorRate()
                .update(input=predicted_strings, target=label_strings)
                .compute()
                .item()
            }
            predictions.metrics.update(additional_metrics)
        assert (
            predictions.loss != None
        ), "Loss is None. Make sure to set loss in ModelOutput"
        self.history.add_batch_metric(
            MetricEntry(predictions.metrics, predictions.loss.cpu().item()),
            (
                DecodedPredictionBatch(
                    predictions=predicted_strings, targets=label_strings
                )
                if self.mode == "test" or self.track_non_test_predictions
                else None
            ),
        )

    def evaluate(self) -> SingleEpochHistory:
        return self.history

    def decode_predictions(
        self, predictions: ModelOutput, sample: SampleBatch
    ) -> DecodedPredictionBatch:
        predicted_ids = predictions.logits.argmax(dim=-1).cpu().numpy()
        predicted_strings = self.tokenizer.batch_decode(
            predicted_ids, group_tokens=True
        )
        label_strings = (
            self.tokenizer.batch_decode(sample.target.cpu().numpy(), group_tokens=False)
            if sample.target is not None
            else None
        )
        return DecodedPredictionBatch(predicted_strings, label_strings)
