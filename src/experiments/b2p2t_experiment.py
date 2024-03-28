from math import nan
import sys
from src.decoding.decoding_types import LLMOutput
from src.model.b2p2t_model import B2P2TModel
from src.model.b2p2t_model import B2P2TModelArgsModel
from src.datasets.brain2text_w_phonemes import (
    Brain2TextWPhonemesDataset,
)
from src.datasets.batch_types import PhonemeSampleBatch
from src.model.b2tmodel import B2TModel, ModelOutput
from src.args.base_args import (
    B2TDatasetArgsModel,
    BaseExperimentArgsModel,
)
from src.experiments.experiment import Experiment
from src.args.yaml_config import YamlConfigModel
from typing import IO, Literal, cast
from torch.utils.data import DataLoader
import torch
from abc import abstractmethod
import torch
import numpy as np
from edit_distance import SequenceMatcher
import pickle
import uuid
import os
import subprocess
from shutil import rmtree
from src.train.evaluator import Evaluator
from src.train.history import MetricEntry, SingleEpochHistory, DecodedPredictionBatch
import json


class B2P2TEvaluator(Evaluator):
    def __init__(self, decoding_script: str, mode: Literal["train", "val", "test"]):
        super().__init__(mode)
        self.temp_dir = f"temp/{uuid.uuid4()}"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.file_order: list[str] = []
        self.losses: list[float] = []
        self.metrics: list[dict[str, float]] = []
        self.decoding_script = decoding_script

    def _track_batch(self, predictions: ModelOutput, sample: PhonemeSampleBatch):
        if self.mode == "test":
            filename = f"{self.n_losses}.pkl"
            with open(f"{self.temp_dir}/{filename}", "wb") as handle:
                pickle.dump((sample, predictions), handle)
                self.file_order.append(filename)
        assert predictions.loss != None, "Loss is None."
        self.losses.append(predictions.loss.item())
        predictions.metrics.update(
            phoneme_error_rate=self._calc_phoneme_error_rate(sample, predictions)
        )
        self.metrics.append(predictions.metrics)

    def evaluate(self) -> SingleEpochHistory:
        if self.mode == "test":
            new_env = os.environ.copy()
            new_env["PYTHONPATH"] = "/hpi/fs00/home/tobias.fiedler/brain2text"
            new_env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            print(
                "[B2P2T Evaluator] Running external script for decoding (this might take a while)"
            )
            process = subprocess.Popen(
                ["stdbuf", "-oL"]
                + [
                    "conda",
                    "run",
                    "-n",
                    "lm_decoder",
                    "python",
                    self.decoding_script,
                    "--data_dir",
                    self.temp_dir,
                ],
                env=new_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Read the output line by line as soon as it is available
            while True:
                output = cast(IO[str], process.stdout).readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                sys.stdout.flush()

            out, err = process.communicate()
            if out:
                print(out.strip(), flush=True)
            if err:
                print(err.strip(), file=sys.stderr, flush=True)
            print(
                f"[B2P2T Evaluator] Finished decoding script with return code {process.returncode}"
            )
            if process.returncode != 0:
                raise Exception(f"Error running postprocess_baseline.py")

        out_dir = os.path.join(self.temp_dir, "out")

        history = SingleEpochHistory()
        for i in range(len(self.losses)):

            base_metrics = self.metrics[i]
            loss = self.losses[i]

            decoded = None
            if self.mode == "test":
                filename = self.file_order[i]
                with open(os.path.join(out_dir, filename), "rb") as handle:
                    batch: LLMOutput = pickle.load(handle)
                    if batch.wer_95_confidence_interval != None:
                        wer_CI_start, wer_CI_end = batch.wer_95_confidence_interval
                        base_metrics.update(
                            decoder_wer_CI_start=wer_CI_start,
                            decoder_wer_CI_end=wer_CI_end,
                        )
                    if batch.cer_95_confidence_interval != None:
                        cer_CI_start, cer_CI_end = batch.cer_95_confidence_interval
                        base_metrics.update(
                            decoder_cer_CI_start=cer_CI_start,
                            decoder_cer_CI_end=cer_CI_end,
                        )
                    base_metrics.update(decoder_wer=batch.wer, decoder_cer=batch.cer)
                    decoded = DecodedPredictionBatch(
                        batch.decoded_transcripts, batch.target_transcripts
                    )
            history.add_batch_metric(MetricEntry(base_metrics, loss), decoded)
        return history

    def clean_up(self):
        rmtree(self.temp_dir)

    def _calc_phoneme_error_rate(
        self, batch: PhonemeSampleBatch, predictions: ModelOutput
    ):
        adjustedLens = predictions.logit_lens
        assert adjustedLens != None, "logit_lens is None."
        if batch.target == None or batch.target_lens == None:
            return nan
        pred = predictions.logits
        total_edit_distance = 0
        total_seq_length = 0
        for iterIdx in range(pred.shape[0]):
            decodedSeq = torch.argmax(
                torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                dim=-1,
            )  # [num_seq,]
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])
            trueSeq = np.array(
                batch.target[iterIdx][0 : batch.target_lens[iterIdx]].cpu().detach()
            )
            matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
            dist = matcher.distance()
            if dist == None:
                print(
                    "[evaluate batch]: distance from sequence matcher is None, skipping."
                )
                continue
            total_edit_distance += dist
            total_seq_length += len(trueSeq)

        return total_edit_distance / total_seq_length


class B2P2TArgsModel(BaseExperimentArgsModel, B2TDatasetArgsModel, B2P2TModelArgsModel):
    decoding_script: str = "src/decoding/postprocess_baseline.py"


class B2P2TExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def get_args_model():
        return B2P2TArgsModel

    def _create_model(self):
        return B2P2TModel(self.config, self._create_neural_decoder())

    @abstractmethod
    def _create_neural_decoder(self) -> B2TModel:
        raise NotImplementedError()

    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        return Brain2TextWPhonemesDataset(
            config=self.config,
            yaml_config=self.yaml_config,
            split=split,
        )

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        ds = self._create_dataset(split)
        return DataLoader(
            self._create_dataset(split),
            batch_size=self.base_config.batch_size,
            shuffle=split != "test" or not self.config.competition_mode,
            collate_fn=ds.get_collate_fn(),
        )

    def get_vocab(self) -> list[str]:
        return Brain2TextWPhonemesDataset.vocab

    def _get_in_size_after_preprocessing(self):
        return (256) * self.config.unfolder_kernel_len

    def create_evaluator(self, mode: Literal["train", "val", "test"]) -> Evaluator:
        return B2P2TEvaluator(self.config.decoding_script, mode)

    def process_test_results(self, test_results: SingleEpochHistory):
        worst_wer = 0
        best_wer = 10
        best_i = -1
        worst_i = -1
        perfect: list[dict] = []
        total = len(test_results.metrics)

        def get_batch(i: int):
            entry = test_results.decoded[i]
            if entry is not None:
                return {
                    "predictions": entry.predictions,
                    "targets": entry.targets,
                }
            return {}

        for i, metric in enumerate(test_results.metrics):
            wer = metric.metrics["decoder_wer"]
            if wer < best_wer:
                best_wer = wer
                best_i = i
            if wer > worst_wer:
                worst_wer = wer
                worst_i = i
            if wer == 0:
                perfect.append(get_batch(i))
        with open(os.path.join(self.results_dir, "best_worst.json"), "w") as f:
            json.dump(
                {
                    "best": {
                        "metrics": test_results.metrics[best_i].metrics,
                        "batch": get_batch(best_i),
                    },
                    "worst": {
                        "metrics": test_results.metrics[worst_i].metrics,
                        "batch": get_batch(worst_i),
                    },
                    "perfect": {
                        "count": len(perfect),
                        "items": perfect,
                    },
                    "total": total,
                },
                f,
                indent=5,
            )

        if self.config.competition_mode == True:
            with open(os.path.join(self.results_dir, "submission.txt"), "w") as f:
                for batch in test_results.decoded:
                    assert batch is not None
                    for pred in batch.predictions:
                        f.write(f"{pred}\n")
