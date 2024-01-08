from typing import Any, Literal
from torch.nn.functional import pad
from src.args.b2t_resnet_args import B2TResnetArgsModel
import torch
import re
from src.args.base_args import B2TDatasetArgsModel
from src.args.yaml_config import YamlConfigModel
import os
from src.model.b2tmodel import B2TModel
from src.datasets.brain2text import Brain2TextDataset
from src.experiments.experiment import Experiment
from src.model.b2t_resnet_model import B2TResnetModel
from transformers import AutoTokenizer
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset


class B2TResnetExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        base_dir = os.path.join(yamlConfig.cache_dir, "audio")
        cache_dir = os.path.join(base_dir, "cache")
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        self.config = B2TResnetArgsModel(**config)
        self.ds_config = B2TDatasetArgsModel(**config)
        super().__init__(config, yamlConfig)
        self.model: B2TResnetModel = self.model

    def get_name(self) -> str:
        return "b2t_resnet"

    @staticmethod
    def get_args_model():
        return B2TResnetArgsModel

    def _create_tokenizer(self):
        # TODO: do we need own tokenizer for this?
        if self.config.tokenizer == "wav2vec_pretrained":
            return AutoTokenizer.from_pretrained(
                self.config.wav2vec_checkpoint,
                cache_dir=self.yaml_config.cache_dir,
            )
        raise Exception(f"Tokenizer {self.config.tokenizer} not supported yet")

    def _create_model(self) -> B2TModel:
        return B2TResnetModel(
            self.config, self.tokenizer.vocab_size, self.tokenizer.pad_token_id
        )

    def get_collate_fn(self):
        def _collate(batch: list[tuple[torch.Tensor, str]]):
            max_audio_len = max([x.size(0) for x, _ in batch])
            padded_audio = [
                pad(
                    x,
                    (0, 0, 0, max_audio_len - x.size(0)),
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

            return torch.stack(padded_audio), batch_label_ids

        return _collate

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "classifier":
                # TODO: return correct parameters
                return self.model.resnet.fc.parameters()
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
        return Brain2TextDataset(self.ds_config, self.yaml_config, split)
