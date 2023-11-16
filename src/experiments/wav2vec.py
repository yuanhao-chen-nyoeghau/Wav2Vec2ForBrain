from src.experiments.experiment import Experiment
from argparse import ArgumentParser
from src.args.base_args import BaseArgsModel


class Wav2VecArgsModel(BaseArgsModel):
    pass


class Wav2VecExperiment(Experiment):
    def __init__(self, config: dict):
        self.config = Wav2VecArgsModel(**config)

    def run(self):
        print(self.config)

    def get_name(self) -> str:
        return "wav2vec"

    @staticmethod
    def get_args_model():
        return Wav2VecArgsModel
