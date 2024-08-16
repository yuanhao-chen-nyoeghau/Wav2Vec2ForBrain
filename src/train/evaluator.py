from edit_distance import SequenceMatcher
from math import nan
from typing import Literal, cast
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
from transformers import Wav2Vec2ProcessorWithLM
from src.datasets.timit_ctc_dataset import TimitAudioSeqDataset, TimitSeqSampleBatch
import torch
import numpy as np
from src.util.phoneme_helper import PHONE_DEF_SIL


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


class EnhancedDecodedBatch(DecodedPredictionBatch):
    predictions_lm_decoded: list[str]


class EvaluatorWithW2vLMDecoder(DefaultEvaluator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mode: Literal["train", "val", "test"],
        cache_dir: str,
        processor_checkpoint: str,
        track_non_test_predictions: bool = False,
        lm_decode_test_predictions: bool = False,
    ):
        super().__init__(tokenizer, mode, track_non_test_predictions)
        self.history = SingleEpochHistory()
        self.tokenizer = tokenizer

        # Tutorial: https://huggingface.co/blog/wav2vec2-with-ngram
        self.processor = (
            Wav2Vec2ProcessorWithLM.from_pretrained(
                processor_checkpoint, cache_dir=cache_dir
            )
            if lm_decode_test_predictions and mode == "test"
            else None
        )

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
        decoded_batch = EnhancedDecodedBatch(
            predictions=predicted_strings, targets=label_strings
        )

        if label_strings is not None:
            additional_metrics = {
                "word_error_rate": WordErrorRate()
                .update(input=predicted_strings, target=label_strings)
                .compute()
                .item()
            }
            if self.processor is not None and self.mode == "test":
                processed = self.processor.batch_decode(
                    predictions.logits.detach().cpu().numpy()
                )
                additional_metrics["word_error_rate_lm_decode"] = (
                    WordErrorRate()
                    .update(input=cast(list[str], processed.text), target=label_strings)
                    .compute()
                    .item()
                )
                decoded_batch.predictions_lm_decoded = cast(list[str], processed.text)

            predictions.metrics.update(additional_metrics)

        assert (
            predictions.loss != None
        ), "Loss is None. Make sure to set loss in ModelOutput"

        self.history.add_batch_metric(
            MetricEntry(predictions.metrics, predictions.loss.cpu().item()),
            (
                decoded_batch
                if self.mode == "test" or self.track_non_test_predictions
                else None
            ),
        )

class TimitDecodedBatch(DecodedPredictionBatch):
    original_targets: list[str]
class TimitSeqW2VSUCEvaluator(Evaluator):
    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        super().__init__(mode, track_non_test_predictions)
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
            (
                prediction_batch
                if self.mode == "test" or self.track_non_test_predictions
                else None
            ),  # prevent 1GB history files
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

        decoded = TimitDecodedBatch(predicted, labels)
        decoded.original_targets = batch.transcripts

        return total_edit_distance / total_seq_length, decoded
