from typing import Literal, cast
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


class B2P2TGruAndW2VArgsModel(
    B2P2TArgsModel, B2PSUCArgsModel, W2VBrainEncoderModelArgs
):
    brain_encoder_path: str


class B2P2TGruAndW2VExperiment(B2P2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

        self.tokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(
                "facebook/wav2vec2-base-960h",
                cache_dir=self.yaml_config.cache_dir,
            ),
        )

    def get_name(self) -> str:
        return "b2p2t_gru+w2v"

    @staticmethod
    def get_args_model():
        return B2P2TGruAndW2VArgsModel

    def _create_model(self):
        brain_encoder = DiscriminatorDataset.brain_feature_extractor_from_config(
            self.config, self.config.brain_encoder_path
        )
        model = W2VBrainEncoderModel(self.config, brain_encoder)
        return model

    def create_evaluator(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        return DefaultEvaluator(self.tokenizer, mode, track_non_test_predictions)

    def process_test_results(self, test_results: SingleEpochHistory):
        pass

    def _create_neural_decoder(self):
        pass
