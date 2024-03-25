from typing import Any
from src.experiments.b2t_wav2vec_experiment import B2TWav2VecExperiment
from src.args.b2t_resnet_args import B2TWav2VecResnetArgsModel
from src.args.yaml_config import YamlConfigModel
from src.model.b2tmodel import B2TModel
from src.model.b2t_resnet_model import (
    B2TCustomEncoderW2VFineTuningModel,
    B2TCustomEncoderW2VPretrainingModel,
)
from torch.optim.optimizer import Optimizer


class B2TWav2VecResnetExperiment(B2TWav2VecExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        super().__init__(config, yamlConfig)
        self.config = B2TWav2VecResnetArgsModel(**config)
        self.model: (
            B2TCustomEncoderW2VPretrainingModel | B2TCustomEncoderW2VFineTuningModel
        ) = self.model

    def get_name(self) -> str:
        return "b2t_wav2vec_resnet"

    @staticmethod
    def get_args_model():
        return B2TWav2VecResnetArgsModel

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "wav2vec2featureextractor_ours":
                return [
                    {
                        "params": self.model.wav2vec2.wav2vec2.feature_extractor.parameters()
                    },
                ]
            if (
                self.config.unfreeze_strategy
                == "wav2vec2featureextractor_wav2vec2classifier_ours"
            ):
                return [
                    {
                        "params": self.model.wav2vec2.wav2vec2.feature_extractor.parameters()
                    },
                    {"params": self.model.wav2vec2.wav2vec2.head.parameters()},
                ]
            if self.config.unfreeze_strategy == "lm_head":
                return self.model.wav2vec2.lm_head.parameters()
            if self.config.unfreeze_strategy == "all":
                return self.model.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for wav2vec experiment"
            )

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)

    def _create_model(self) -> B2TModel:
        if self.config.mode == "pretraining":
            return B2TCustomEncoderW2VPretrainingModel(
                self.config, self.yaml_config, self.tokenizer
            )
        elif self.config.mode == "finetuning":
            return B2TCustomEncoderW2VFineTuningModel(
                self.config, self.yaml_config, self.tokenizer
            )
        raise Exception(
            f"Mode {self.config.mode} not supported yet by {self.get_name()}"
        )
