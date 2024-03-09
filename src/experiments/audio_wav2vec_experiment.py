import os
from torch.optim.optimizer import Optimizer
from src.datasets.base_dataset import Sample, SampleBatch
from src.datasets.audio import AudioDataset
from src.model.audio_wav2vec_model import AudioWav2VecModel
from src.experiments.experiment import Experiment
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, cast
from src.args.wav2vec_args import AudioWav2VecArgsModel
from transformers import AutoTokenizer
import torch
from torch.nn.functional import pad
import re
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer


class AudioWav2VecExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        base_dir = os.path.join(yamlConfig.cache_dir, "audio")
        cache_dir = os.path.join(base_dir, "cache")
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        self._hugg_dataset = load_dataset(
            "google/fleurs", name="en_us", cache_dir=cache_dir, data_dir=data_dir
        )
        self.config = AudioWav2VecArgsModel(**config)
        self.tokenizer = cast(PreTrainedTokenizer, self._create_tokenizer())
        super().__init__(config, yamlConfig)
        self.model: AudioWav2VecModel = self.model

    def get_name(self) -> str:
        return "audio_wav2vec"

    @staticmethod
    def get_args_model():
        return AudioWav2VecArgsModel

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
        model = AudioWav2VecModel(
            config=self.config,
            yaml_config=self.yaml_config,
            tokenizer=self.tokenizer,
        )
        return model

    def get_collate_fn(self):
        def _collate(batch: list[Sample]):
            max_audio_len = max([x.size(0) for x, _ in batch])
            padded_audio = [
                pad(
                    x,
                    (0, max_audio_len - x.size(0)),
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

            batch_label_ids: torch.Tensor = self.tokenizer(
                [process_label(label) for _, label in batch],
                padding="longest",
                return_tensors="pt",
            ).input_ids

            return SampleBatch(torch.stack(padded_audio), batch_label_ids)

        return _collate

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "wav2vec2featureextractor":
                return [
                    {
                        "params": self.model.wav2vec2.wav2vec2.feature_extractor.parameters()
                    },
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
        return AudioDataset(
            hugg_dataset=cast(DatasetDict, self._hugg_dataset),
            split=split,
        )
