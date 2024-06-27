from src.model.gru_model import GRUModel, GruArgsModel
from src.datasets.brain2text_w_phonemes import (
    Brain2TextWPhonemesDataset,
)
from src.experiments.b2p2t_experiment import B2P2TArgsModel, B2P2TExperiment
from src.args.yaml_config import YamlConfigModel


class B2P2TGruArgsModel(B2P2TArgsModel, GruArgsModel):
    pass


class B2P2TGruExperiment(B2P2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "b2p2t_gru"

    @staticmethod
    def get_args_model():
        return B2P2TGruArgsModel

    def _create_neural_decoder(self):
        return GRUModel(
            self.config,
            Brain2TextWPhonemesDataset.vocab_size,
            self._get_in_size_after_preprocessing(),
        )
