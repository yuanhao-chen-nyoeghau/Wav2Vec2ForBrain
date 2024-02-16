from src.datasets.ctc_text_dataset import CTCTextDataset
from src.experiments.experiment import Experiment
from src.args.wav2vec_args import ACTIVATION_FUNCTION
from src.args.base_args import (
    BaseExperimentArgsModel,
    CTCTextDatasetArgsModel,
)
from src.model.b2tmodel import B2TModel
from src.args.yaml_config import YamlConfigModel
from typing import Literal, cast
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.functional import pad
import re
from transformers import PreTrainedTokenizer


class CtcLmArgsModel(BaseExperimentArgsModel, CTCTextDatasetArgsModel):
    nhead: int = 8
    num_layers: int = 6
    classifier_hidden_sizes: list[int] = []
    classifier_activation: ACTIVATION_FUNCTION = "gelu"


class CtcLmExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        self.yaml_config = yamlConfig
        self.tokenizer = self._create_tokenizer()
        self.dataset = CTCTextDataset(self.config, yamlConfig, self.tokenizer)

        super().__init__(config, yamlConfig)
        self.model: B2TModel = self.model

        self.config: CtcLmArgsModel = self.config

    def get_name(self) -> str:
        return "ctc_lm_experiment"

    @staticmethod
    def get_args_model():
        return CtcLmArgsModel

    def _create_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        from src.model.ctc_lm_model import CTCLMModel

        model = CTCLMModel(self.config, self.tokenizer)
        return model

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> Dataset:
        return self.dataset.get_split(split)

    def get_collate_fn(self):
        def _collate(batch: list[tuple[torch.Tensor, str]]):
            max_block_len = max([x.size(0) for x, _ in batch])
            padded_blocks = [
                pad(
                    x,
                    (0, 0, 0, max_block_len - x.size(0)),
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

            return torch.stack(padded_blocks), batch_label_ids

        return _collate

    def _create_tokenizer(self) -> PreTrainedTokenizer:
        if self.config.tokenizer == "wav2vec_pretrained":
            assert (
                not self.config.tokenizer_checkpoint is None
            ), "Tokenizer checkpoint (--tokenizer_checkpoint) must be set when using --tokenizer=wav2vec_pretrained"

            return cast(
                PreTrainedTokenizer,
                AutoTokenizer.from_pretrained(
                    self.config.tokenizer_checkpoint,
                    cache_dir=self.yaml_config.cache_dir,
                ),
            )
        raise Exception(f"Tokenizer {self.config.tokenizer} not supported yet")
