from src.experiments.b2t_experiment import B2TExperiment
from src.args.base_args import (
    B2TArgsModel,
    B2TDatasetArgsModel,
    BaseExperimentArgsModel,
)
from src.model.b2tmodel import B2TModel, ModelOutput
from torch.optim.optimizer import Optimizer
from src.datasets.brain2text import B2tSampleBatch, Brain2TextDataset
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
from math import floor, isnan
from torch.nn.functional import log_softmax, softmax


def calc_seq_len(index_seq: torch.Tensor):
    for i in range(len(index_seq)):
        j = len(index_seq) - 1 - i
        if index_seq[j].item() > 0:
            return j + 1
    return 0


class B2TCNNArgsModel(B2TArgsModel):
    conv_kernel_time: list[int] = [5, 3, 3, 3]
    conv_kernel_features: list[int] = [128, 1, 1, 1]  # [5, 5, 3, 118]
    conv_out_channels: list[int] = [256, 256, 512, 32]
    conv_padding: list[Literal["same", "valid", 0]] = [0, 0, 0, 0]
    activation: Literal["gelu"] = "gelu"
    conv_bias: bool = True


class CNNModel(B2TModel):
    def __init__(self, config: B2TCNNArgsModel, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

        self.model = nn.Sequential(
            *[
                (
                    nn.Conv2d(
                        (
                            2
                            if floor(i / 2) == 0
                            else config.conv_out_channels[floor(i / 2) - 1]
                        ),
                        config.conv_out_channels[floor(i / 2)],
                        kernel_size=(
                            config.conv_kernel_time[floor(i / 2)],
                            config.conv_kernel_features[floor(i / 2)],
                        ),
                        stride=(1, 1),
                        bias=config.conv_bias,
                        padding=config.conv_padding[floor(i / 2)],
                    )
                    if i % 2 == 0
                    else nn.GELU()
                )
                for i in range(len(config.conv_out_channels) * 2)
            ]
        )

        self.loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(self, batch: B2tSampleBatch) -> ModelOutput:
        x, targets = batch
        assert targets is not None, "Targets must be set"
        device = targets.device

        out = self.model(x)

        if targets is not None:
            targets = torch.where(
                targets == self.tokenizer.pad_token_id, torch.tensor(-100), targets
            )

        target_lens = torch.tensor([calc_seq_len(seq) for seq in targets])

        # out shape: (batch_size, vocab_size, seq_len, 1)
        out = out.squeeze(-1)
        # out shape: (batch_size, vocab_size, seq_len)
        out = out.transpose(1, 2)
        # out shape: (batch_size, seq_len, vocab_size)
        out_intermediate = out
        out = log_softmax(out, -1)

        # non padded mask
        mask = x != 0
        # seq lens without padding
        in_seq_lens = mask.any(-1).sum(-1).max(-1).values.clamp(max=out.shape[1])

        out = out.transpose(0, 1)
        # out shape: (seq_len, batch_size, vocab_size)

        ctc_loss = self.loss(
            out,
            targets,
            in_seq_lens.to(device),
            target_lens.to(device),
        )
        if isnan(ctc_loss.item()):
            pass
        elif ctc_loss.item() < 0:
            print(
                f"\nWarning: loss is negative, this might be due to prediction lens ({in_seq_lens.tolist()}) being smaller than target lens {target_lens.tolist()}\n"
            )
        # TODO: fix loss is inf bug

        # out_intermediate shape: (batch_size, seq_len, vocab_size)
        prediction_variance = softmax(out_intermediate.std(-2), -1)
        var_loss = prediction_variance.mean(-1).mul(10).add(0.000001).pow(-1).sqrt()
        var_loss = torch.min(var_loss.mean(), torch.tensor(100))

        if isnan(var_loss.item()):
            pass

        return ModelOutput(
            out.transpose(0, 1),
            {"ctc_loss": ctc_loss.item(), "var_loss": var_loss.item()},
            ctc_loss,
        )


class CNNExperiment(B2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)
        self.model: B2TModel = self.model

    def get_name(self) -> str:
        return "direct_cnn"

    @staticmethod
    def get_args_model():
        return B2TCNNArgsModel

    def _create_tokenizer(self):
        if self.config.tokenizer == "wav2vec_pretrained":
            assert (
                not self.config.tokenizer_checkpoint is None
            ), "Tokenizer checkpoint (--tokenizer_checkpoint) must be set when using --tokenizer=wav2vec_pretrained"

            return AutoTokenizer.from_pretrained(
                self.config.tokenizer_checkpoint,
                cache_dir=self.yaml_config.cache_dir,
            )
        raise Exception(f"Tokenizer {self.config.tokenizer} not supported yet")

    def _create_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        model = CNNModel(self.config, self.tokenizer)
        return model

    def get_vocab(self) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(
            list(range(self.tokenizer.vocab_size))
        )
