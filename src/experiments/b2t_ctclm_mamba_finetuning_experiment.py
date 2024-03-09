from src.experiments.b2t_ctclm_finetuning_experiment import (
    B2TCTCFinetuningArgsModel,
    B2tCtcLmModel,
    B2tCtcLmFinetuningExperiment,
)
from src.experiments.b2t_mamba_experiment import B2TMambaArgsModel, MambaModel
from src.model.b2tmodel import B2TModel
from src.args.yaml_config import YamlConfigModel
from typing import Literal, Any
from torch.optim.optimizer import Optimizer


class B2TCTCMambaFinetuningArgsModel(B2TCTCFinetuningArgsModel, B2TMambaArgsModel):
    unfreeze_strategy: Literal["all", "only_lm", "only_tokenembedding"] = "all"


class B2tCtcLmMambaFinetuningExperiment(B2tCtcLmFinetuningExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)
        self.model: B2tCtcLmModel = self.model

        assert (
            self.config.preprocessing == "seperate_zscoring"
        ), "Only seperate_zscoring is currently supported"

    def get_name(self) -> str:
        return "ctc_lm_mamba_finetuning"

    @staticmethod
    def get_args_model():
        return B2TCTCMambaFinetuningArgsModel

    def get_b2t_model(self) -> B2TModel:
        """Get the b2t model, pretrained weights are automatically loaded if lm_checkpoint_path is set in the config"""
        return MambaModel(self.config, self.tokenizer.vocab_size)

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "only_lm":
                if self.config.b2t_learning_rate is not None:
                    print(
                        "Warning: b2t_learning_rate is set but will be ignored because unfreeze_strategy is set to only_lm"
                    )
                return self.model.lm_model.parameters()
            if self.config.unfreeze_strategy == "all":
                return [
                    {
                        "params": self.model.b2t_model.parameters(),
                        "lr": (
                            self.config.b2t_learning_rate
                            if self.config.b2t_learning_rate is not None
                            else self.config.learning_rate
                        ),
                    },
                    {"params": self.model.lm_model},
                ]
            if self.config.unfreeze_strategy == "only_tokenembedding":
                if self.config.b2t_learning_rate is not None:
                    print(
                        "Warning: b2t_learning_rate is set but will be ignored because unfreeze_strategy is set to only_tokenembedding"
                    )

                return self.model.lm_model.tok_embed.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} not supported yet"
            )

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)
