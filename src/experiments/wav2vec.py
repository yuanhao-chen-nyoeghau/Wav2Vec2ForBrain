from src.experiments.experiment import Experiment
from argparse import ArgumentParser
from src.args.base_args import BaseExperimentArgsModel
from src.args.yaml_config import YamlConfigModel


class Wav2VecArgsModel(BaseExperimentArgsModel):
    pass


class Wav2VecExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        super().__init__(config, yamlConfig)
        self.config = Wav2VecArgsModel(**config)

    def get_name(self) -> str:
        return "wav2vec"

    @staticmethod
    def get_args_model():
        return Wav2VecArgsModel
