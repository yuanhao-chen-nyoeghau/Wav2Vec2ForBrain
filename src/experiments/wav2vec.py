from torch.optim.optimizer import Optimizer
from src.experiments.experiment import Experiment
from argparse import ArgumentParser
from src.args.base_args import BaseExperimentArgsModel
from src.args.yaml_config import YamlConfigModel
from typing import Literal
from src.args.wav2vec_args import Wav2VecArgsModel
from src.datasets.tokenizer import get_tokenizer
from transformers import AutoTokenizer
from src.model.b2t_wav2vec import B2TWav2Vec
import torch
from torch.nn.functional import pad
from torch.utils.data import default_collate


class Wav2VecExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = Wav2VecArgsModel(**config)
        super().__init__(config, yamlConfig)
        self.model: B2TWav2Vec = self.model

    def get_name(self) -> str:
        return "wav2vec"

    @staticmethod
    def get_args_model():
        return Wav2VecArgsModel

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
            self.config.loss_function == "ctc",
            "Only ctc loss is currently supported",
        )
        model = B2TWav2Vec(
            config=self.config,
            yaml_config=self.yaml_config,
            tokenizer=self.tokenizer,
        )
        return model

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

            all_labels = torch.eye(self.tokenizer.__len__())
            batch_label_ids: list[list[int]] = self.tokenizer(
                [label.upper() for _, label in batch],
                padding="longest",
                return_tensors="pt",
            ).input_ids

            return torch.stack(padded_blocks), batch_label_ids

        return _collate

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "wav2vec2featureextractor_ours":
                return [
                    {"params": self.model.brain2audioshape.parameters()},
                    {
                        "params": self.model.wav2vec2.wav2vec2.feature_extractor.parameters()
                    },
                ]
            if self.config.unfreeze_strategy == "all":
                return self.model.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for wav2vec experiment"
            )

        return self._get_optimizer_cls()(
            get_trainable_params(), lr=self.config.learning_rate
        )
