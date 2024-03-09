from src.datasets.brain2text_w_phonemes import (
    Brain2TextWPhonemesDataset,
)
from src.experiments.b2p2t_experiment import B2P2TArgsModel, B2P2TExperiment
from src.model.mamba_model import MambaArgsModel, MambaModel
from src.experiments.experiment import Experiment
from src.args.yaml_config import YamlConfigModel


class B2P2TMambaArgsModel(B2P2TArgsModel, MambaArgsModel):
    pass


class B2P2TMambaExperiment(B2P2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "b2p2t_mamba"

    @staticmethod
    def get_args_model():
        return B2P2TMambaArgsModel

    def _create_model(self):
        return MambaModel(self.config, Brain2TextWPhonemesDataset.vocab_size)
