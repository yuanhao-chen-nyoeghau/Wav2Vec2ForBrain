from src.datasets.base_dataset import SampleBatch
from src.datasets.brain2text import B2tSampleBatch
from src.model.ctc_lm_model import CTCLMModel
from src.experiments.ctc_lm_experiment import CtcLmArgsModel
from src.experiments.b2t_experiment import B2TExperiment
from src.args.base_args import (
    B2TArgsModel,
)
from src.model.b2tmodel import B2TModel, ModelOutput
from src.args.yaml_config import YamlConfigModel
from typing import Literal, Optional, Any
import torch
from torch import nn

from abc import abstractmethod
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedTokenizer


class B2TCTCFinetuningArgsModel(CtcLmArgsModel, B2TArgsModel):
    lm_checkpoint_path: Optional[str] = None
    b2t_checkpoint_path: Optional[str] = None
    unfreeze_strategy: Literal["all", "only_lm"] = "all"
    b2t_learning_rate: Optional[float] = None


class B2tCtcLmModel(B2TModel):
    def __init__(
        self,
        config: B2TCTCFinetuningArgsModel,
        b2t_model: B2TModel,
        lm_model: CTCLMModel,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.config = config
        self.lm_model = lm_model
        self.b2t_model = b2t_model
        self.tokenizer = tokenizer
        self.loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(self, batch: B2tSampleBatch) -> ModelOutput:
        x, targets = batch
        assert targets is not None, "Targets must be set"
        if targets is not None:
            targets = torch.where(
                targets == self.tokenizer.pad_token_id, torch.tensor(-100), targets
            )

        b2t_out = self.b2t_model.forward(batch)
        lm_out = self.lm_model.forward(SampleBatch(b2t_out.logits.softmax(-1), targets))
        return lm_out


class B2tCtcLmFinetuningExperiment(B2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)
        self.model: B2tCtcLmModel = self.model

        assert (
            self.config.preprocessing == "seperate_zscoring"
        ), "Only seperate_zscoring is currently supported"

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError("get_name must be implemented in a subclass")

    @staticmethod
    def get_args_model():
        return B2TCTCFinetuningArgsModel

    def get_lm_model(self) -> CTCLMModel:
        """Get the language model model to be used for finetuning, pretrained weights are automatically loaded if lm_checkpoint_path is set in the config"""
        return CTCLMModel(self.config, self.tokenizer)

    @abstractmethod
    def get_b2t_model(self) -> B2TModel:
        """Get the b2t model, pretrained weights are automatically loaded if lm_checkpoint_path is set in the config"""
        raise NotImplementedError("get_b2t_model must be implemented in a subclass")

    def _create_model(self) -> B2tCtcLmModel:
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        lm_model = self.get_lm_model()
        if self.config.lm_checkpoint_path is not None:
            print("Loading lm model from", self.config.lm_checkpoint_path)
            lm_model.load_state_dict(torch.load(self.config.lm_checkpoint_path))
        b2t_model = self.get_b2t_model()
        if self.config.b2t_checkpoint_path is not None:
            print("Loading b2t model from", self.config.b2t_checkpoint_path)
            b2t_model.load_state_dict(torch.load(self.config.b2t_checkpoint_path))

        model = B2tCtcLmModel(self.config, b2t_model, lm_model, self.tokenizer)
        return model

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

            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} not supported yet"
            )

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)
