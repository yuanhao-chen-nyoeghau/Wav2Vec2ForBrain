from src.decoding.postprocess import LLMOutput
from src.model.b2p2t_model import B2P2TModel
from src.model.b2p2t_model import B2P2TModelArgsModel
from src.datasets.brain2text_w_phonemes import (
    Brain2TextWPhonemesDataset,
    decode_predicted_phoneme_ids,
)
from src.datasets.batch_types import PhonemeSampleBatch
from src.model.b2tmodel import B2TModel, ModelOutput
from src.args.base_args import (
    B2TDatasetArgsModel,
    BaseExperimentArgsModel,
)
from src.experiments.experiment import DecodedPredictionBatch, Experiment
from src.args.yaml_config import YamlConfigModel
from typing import Literal
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


class B2P2TArgsModel(BaseExperimentArgsModel, B2TDatasetArgsModel, B2P2TModelArgsModel):
    pass


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

    def decode_predictions(
        self, predictions: ModelOutput, sample: PhonemeSampleBatch
    ) -> DecodedPredictionBatch:
        predicted_ids = predictions.logits.argmax(dim=-1).cpu().numpy()
        # tuple[PhonemeSampleBatch, ModelOutput]
        temp_dir = f"temp/{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)

        filename = "1.pkl"
        with open(f"{temp_dir}/{filename}", "wb") as handle:
            pickle.dump((sample, predictions), handle)

        # execute: conda run -n ${CONDA_ENV_NAME} python script.py
        result = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "lm_decoder",
                "python",
                "src/decoding/postprocess.py",
                "--data_dir",
                temp_dir,
            ]
        )
        if result.returncode != 0:
            raise Exception(f"Error running postprocess.py: {result.stderr}")
        out_file = os.path.join(temp_dir, "out", filename)
        with open(out_file, "rb") as handle:
            batch: LLMOutput = pickle.load(handle)

        rmtree(temp_dir)
        # TODO: decode phonemes to strings
        predicted_strings = batch.decoded_transcripts

        label_strings = sample.transcriptions
        return DecodedPredictionBatch(predicted_strings, label_strings)

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
            shuffle=True,
            collate_fn=ds.get_collate_fn(),
        )

    def get_vocab(self) -> list[str]:
        return Brain2TextWPhonemesDataset.vocab

    def _get_in_size_after_preprocessing(self):
        return (256) * self.config.unfolder_kernel_len

    def evaluate_batch(
        self, batch: PhonemeSampleBatch, predictions: ModelOutput
    ) -> dict[str, float]:
        metrics = super().evaluate_batch(batch, predictions)

        # compute phoneme error rate. Source: https://github.com/cffan/neural_seq_decoder/blob/master/src/neural_decoder/neural_decoder_trainer.py#L172
        adjustedLens = (
            (batch.input_lens - self.config.unfolder_kernel_len)
            / self.config.unfolder_stride_len
        ).to(torch.int32)

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

        metrics["phoneme_error_rate"] = total_edit_distance / total_seq_length
        return metrics
