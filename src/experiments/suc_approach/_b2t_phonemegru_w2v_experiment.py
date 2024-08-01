from typing import Any, Literal, cast

import torch
from src.datasets.brain2text_w_phonemes import Brain2TextWPhonemesDataset
from src.experiments.b2p2t_experiment import B2P2TArgsModel
from src.model.b2p2t_model import DEFAULT_UNFOLDER_KERNEL_LEN, B2P2TModel
from src.model.gru_model import (
    GRUModelWithLinearProject,
    GRUModelWithLinearProjectArgs,
)
from src.experiments.b2t_experiment import B2TArgsModel, B2TExperiment
from src.model.w2v_custom_feat_extractor import (
    W2VBrainEncoderModel,
    W2VBrainEncoderModelArgs,
)
from src.args.yaml_config import YamlConfigModel
from torch.optim.optimizer import Optimizer


class B2TPhonemeGruAndW2VArgsModel(
    B2TArgsModel,
    GRUModelWithLinearProjectArgs,
    W2VBrainEncoderModelArgs,
    B2P2TArgsModel,
):
    b2p2t_gru_checkpoint: str
    unfreeze_strategy: Literal["gru_project"] = "gru_project"


class B2TPhonemeGruAndW2VExperiment(B2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "b2p2t_phonemegru+w2v"

    @staticmethod
    def get_args_model():
        return B2TPhonemeGruAndW2VArgsModel

    def _create_model(self):
        brain_encoder = B2P2TModel(
            self.config,
            GRUModelWithLinearProject(
                self.config,
                Brain2TextWPhonemesDataset.vocab_size,
                B2P2TModel.get_in_size_after_preprocessing(DEFAULT_UNFOLDER_KERNEL_LEN),
            ),
        )

        state = torch.load(self.config.b2p2t_gru_checkpoint, map_location="cuda")

        adapted_state = {}
        for key in state.keys():
            # Move keys as we have a wrapper "gru" now where there was just neural_decoder before
            adapted_key = (
                "neural_decoder.gru." + key[len("neural_decoder.") :]
                if key.startswith("neural_decoder.")
                else key
            )
            adapted_state[adapted_key] = state[key]
        brain_encoder.load_state_dict(adapted_state, strict=False)
        model = W2VBrainEncoderModel(self.config, brain_encoder)
        return model

    def create_optimizer(self) -> Optimizer:
        optim_cls: Any = self._get_optimizer_cls()
        assert self.config.unfreeze_strategy == "gru_project"
        return optim_cls(
            cast(
                GRUModelWithLinearProject,
                cast(
                    B2P2TModel, cast(W2VBrainEncoderModel, self.model).brain_encoder
                ).neural_decoder,
            ).linear_project.parameters(),
            lr=self.base_config.learning_rate,
            weight_decay=self.base_config.weight_decay,
            eps=self.base_config.optimizer_epsilon,
        )
