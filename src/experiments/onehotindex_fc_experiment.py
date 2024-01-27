from src.args.base_args import B2TDatasetArgsModel, BaseExperimentArgsModel
from src.model.b2tmodel import B2TModel, ModelOutput
from torch.optim.optimizer import Optimizer
from src.datasets.brain2text import Brain2TextDataset
from src.experiments.experiment import Experiment
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, Optional, cast
from src.args.wav2vec_args import B2TWav2VecArgsModel
from transformers import AutoTokenizer
from src.model.b2t_wav2vec_model import B2TWav2Vec
import torch
from torch.nn.functional import pad
import re
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from math import floor


def calc_seq_len(index_seq: torch.Tensor):
    for i in range(len(index_seq)):
        j = len(index_seq) - 1 - i
        if index_seq[j].item() > 0:
            return j + 1


class FCModel(B2TModel):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1000, tokenizer.vocab_size), nn.LogSoftmax(dim=-1)
        )
        self.one_hot_matrix = torch.eye(1000)
        self.loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.tokenizer = tokenizer

    def forward(
        self, _x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        assert targets is not None, "Targets must be set"
        device = targets.device
        seq_len = targets.shape[-1]
        batch_size = targets.shape[0]
        single = torch.stack(
            [
                self.one_hot_matrix[floor(i / 2) if i % 2 == 0 else 999]
                for i in range(seq_len * 2)
            ]
        ).unsqueeze(0)
        batch = single.repeat(batch_size, 1, 1).to(device)
        out = self.model(batch)

        if targets is not None:
            targets = torch.where(
                targets == self.tokenizer.pad_token_id, torch.tensor(-100), targets
            )

        predicted_indices = out.argmax(-1)

        predicted_lens = torch.tensor([calc_seq_len(seq) for seq in predicted_indices])

        target_lens = torch.tensor([calc_seq_len(seq) for seq in targets])

        loss = self.loss(
            out.transpose(0, 1),
            targets,
            predicted_lens.to(device),
            target_lens.to(device),
        )
        # TODO: fix loss is inf bug

        return ModelOutput(out, {"ctc_loss": loss.item()}, loss)


class OneHotArgsModel(BaseExperimentArgsModel, B2TDatasetArgsModel):
    pass


class OneHotIndexExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)
        self.model: B2TModel = self.model

    def get_name(self) -> str:
        return "onehotindex_fc"

    @staticmethod
    def get_args_model():
        return OneHotArgsModel

    def _create_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        model = FCModel(self.tokenizer)
        return model
