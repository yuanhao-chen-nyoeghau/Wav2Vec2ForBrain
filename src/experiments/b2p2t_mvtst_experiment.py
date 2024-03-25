from src.experiments.mvts_transformer_experiment import B2TMvtsTransformerArgsModel
from src.model.mvts_transformer_model import MvtsTransformerModel
from src.datasets.brain2text_w_phonemes import (
    Brain2TextWPhonemesDataset,
)
from src.experiments.b2p2t_experiment import B2P2TArgsModel, B2P2TExperiment
from src.args.yaml_config import YamlConfigModel


class B2P2TMvtstArgsModel(B2P2TArgsModel, B2TMvtsTransformerArgsModel):
    pass


class B2P2TMvtstExperiment(B2P2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "b2p2t_mvtst"

    @staticmethod
    def get_args_model():
        return B2P2TMvtstArgsModel

    def _create_neural_decoder(self):
        return MvtsTransformerModel(
            self.config, 
            Brain2TextWPhonemesDataset.vocab_size,
            self._get_in_size_after_preprocessing(),
        )
