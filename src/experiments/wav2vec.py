from src.experiments.experiment import Experiment
from argparse import ArgumentParser
from src.args.base_args import BaseExperimentArgsModel
from src.args.yaml_config import YamlConfigModel
from typing import Literal
from src.args.wav2vec_args import Wav2VecArgsModel
from src.datasets.tokenizer import get_tokenizer
from transformers import AutoTokenizer
from src.model.b2t_wav2vec import B2TWav2Vec


class Wav2VecExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = Wav2VecArgsModel(**config)
        super().__init__(config, yamlConfig)

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

    def get_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        return B2TWav2Vec(
            config=self.config,
            yaml_config=self.yaml_config,
        )
