import os
from torch.optim.optimizer import Optimizer
from src.args.b2t_audio_args import B2TAudioDatasetArgsModel, B2TAudioWav2VecArgsModel
from src.model.b2t_audio_wav2vec_model import B2TAudioWav2VecModel
from src.datasets.b2t_audio import B2TAudioDataset
from src.model.audio_wav2vec_model import AudioWav2VecModel
from src.experiments.experiment import Experiment
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal
from transformers import AutoTokenizer
import torch
from torch.nn.functional import pad
import re
from torch.utils.data import Dataset
from src.args.base_args import B2TDatasetArgsModel


class B2TAudioWav2VecExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = B2TAudioWav2VecArgsModel(**config)
        self.ds_config = B2TAudioDatasetArgsModel(**config)

        super().__init__(config, yamlConfig)
        self.model: B2TAudioWav2VecModel = self.model

    def get_name(self) -> str:
        return "b2t_audio_wav2vec"

    @staticmethod
    def get_args_model():
        return B2TAudioWav2VecArgsModel

    def _create_tokenizer(self):
        if self.config.tokenizer == "wav2vec_pretrained":
            return AutoTokenizer.from_pretrained(
                self.config.wav2vec_checkpoint,
                cache_dir=self.yaml_config.cache_dir,
            )
        raise Exception(f"Tokenizer {self.config.tokenizer} not supported yet")

    def _create_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )

        model = B2TAudioWav2VecModel(self.config, self.yaml_config, self.tokenizer)
        return model

    def get_collate_fn(self):
        def _collate(batch: list[tuple[torch.Tensor, str]]):
            max_audio_len = max([x.size(0) for x, _ in batch])
            padded_audio = [
                pad(
                    x,
                    (0, max_audio_len - x.size(0))
                    if self.ds_config.mean_reduction_data
                    else (
                        0,
                        0,
                        0,
                        max_audio_len - x.size(0),
                    ),
                    mode="constant",
                    value=0,
                )
                for x, _ in batch
            ]

            def process_label(label: str) -> str:
                if self.config.remove_punctuation:
                    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'
                    label = re.sub(chars_to_ignore_regex, "", label)
                # label = label.upper()
                return label

            batch_label_ids: list[list[int]] = self.tokenizer(
                [process_label(label) for _, label in batch],
                padding="longest",
                return_tensors="pt",
            ).input_ids
            batched_inputs = torch.stack(padded_audio)
            return batched_inputs, batch_label_ids

        return _collate

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "wav2vec2featureextractor":
                return [
                    {
                        "params": self.model.wav2vec2.wav2vec2.feature_extractor.parameters()
                    },
                    {"params": self.model.summarizer_module.parameters()},
                ]
            if self.config.unfreeze_strategy == "all":
                return self.model.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for wav2vec experiment"
            )

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> Dataset:
        return B2TAudioDataset(
            config=self.ds_config,
            yaml_config=self.yaml_config,
            split=split,
        )
