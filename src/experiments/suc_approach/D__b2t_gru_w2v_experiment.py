from typing import Literal, cast
from src.experiments.b2t_experiment import B2TArgsModel, B2TExperiment
from src.datasets.discriminator_dataset import DiscriminatorDataset
from src.model.w2v_custom_feat_extractor import (
    W2VBrainEncoderModel,
    W2VBrainEncoderModelArgs,
)
from src.model.b2p_suc import B2PSUCArgsModel
from src.experiments.b2p2t_experiment import B2P2TArgsModel, B2P2TExperiment
from src.args.yaml_config import YamlConfigModel
from src.train.evaluator import DefaultEvaluator
from src.train.history import SingleEpochHistory
from transformers import AutoTokenizer, PreTrainedTokenizer


class B2TGruAndW2VArgsModel(B2TArgsModel, B2PSUCArgsModel, W2VBrainEncoderModelArgs):
    brain_encoder_path: str


class B2TGruAndW2VExperiment(B2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "b2p2t_gru+w2v"

    @staticmethod
    def get_args_model():
        return B2TGruAndW2VArgsModel

    def _create_model(self):
        brain_encoder = DiscriminatorDataset.brain_feature_extractor_from_config(
            self.config, self.config.brain_encoder_path
        )
        model = W2VBrainEncoderModel(self.config, brain_encoder)
        return model
