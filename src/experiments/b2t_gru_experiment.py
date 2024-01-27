from unicodedata import bidirectional
from src.experiments.b2t_experiment import B2TExperiment
from src.args.base_args import (
    B2TArgsModel,
)
from src.model.b2tmodel import B2TModel, ModelOutput
from src.args.yaml_config import YamlConfigModel
from typing import Optional
import torch
from torch import nn
from transformers import PreTrainedTokenizer
from math import isnan
from torch.nn.functional import log_softmax
from typing import Literal
from torch.nn import Linear


def calc_seq_len(index_seq: torch.Tensor):
    for i in range(len(index_seq)):
        j = len(index_seq) - 1 - i
        if index_seq[j].item() > 0:
            return j + 1
    return 0


class B2TGruArgsModel(B2TArgsModel):
    hidden_size: int = 256
    bidirectional: bool = True
    num_gru_layers: int = 2
    bias: bool = True
    dropout: float = 0.0
    learnable_inital_state: bool = False
    classifier_hidden_sizes: list[int] = [256, 128, 64]
    classifier_activation: Literal["gelu"] = "gelu"


class GRUModel(B2TModel):
    def __init__(self, config: B2TGruArgsModel, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.num_directions = 2 if config.bidirectional else 1

        self.hidden_start = nn.Parameter(
            torch.randn(
                self.num_directions * config.num_gru_layers,
                config.hidden_size,
                requires_grad=True,
            )
        )
        self.gru = torch.nn.GRU(
            256,
            config.hidden_size,
            config.num_gru_layers,
            dropout=config.dropout,
            bias=config.bias,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        classifier_layers = []
        for i in range(-1, len(config.classifier_hidden_sizes)):
            is_last = i + 1 == len(config.classifier_hidden_sizes)
            is_first = i == -1

            in_size = (
                config.hidden_size * self.num_directions
                if is_first
                else config.classifier_hidden_sizes[i]
            )
            out_size = (
                tokenizer.vocab_size
                if is_last
                else config.classifier_hidden_sizes[i + 1]
            )

            classifier_layers.append(Linear(in_size, out_size))
            if not is_last:
                classifier_layers.append(nn.GELU())

        self.classifier = nn.Sequential(*classifier_layers)
        self.loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        assert targets is not None, "Targets must be set"
        device = targets.device
        if targets is not None:
            targets = torch.where(
                targets == self.tokenizer.pad_token_id, torch.tensor(-100), targets
            )

        target_lens = torch.tensor([calc_seq_len(seq) for seq in targets])

        # TODO: look into: The input can also be a packed variable length sequence.
        # See torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence() for details.

        # TODO: make hidden start only for one sample and repeat times batch size here
        out, h_out = (
            self.gru(
                x, self.hidden_start.unsqueeze(1).repeat(1, self.config.batch_size, 1)
            )
            if self.config.learnable_inital_state
            else self.gru(x)
        )

        # out shape: (batch_size, seq_len, hidden_size * num_directions)
        out = self.classifier(out)

        # out shape: (batch_size, seq_len, vocab_size)

        out = log_softmax(out, -1)

        # non padded mask
        mask = x != 0
        # seq lens without padding
        # mask shape: (batch_size, seq_len, 256)
        in_seq_lens = mask.any(-1)
        # in_seq_lens shape: (batch_size, seq_len)
        in_seq_lens = in_seq_lens.sum(-1)
        # in_seq_lens shape: (batch_size)
        in_seq_lens = in_seq_lens.clamp(max=out.shape[1])

        out = out.transpose(0, 1)
        # out shape: (seq_len, batch_size, vocab_size)

        ctc_loss = self.loss(
            out,
            targets,
            in_seq_lens.to(device),
            target_lens.to(device),
        )

        if ctc_loss.item() < 0:
            print(
                f"\nWarning: loss is negative, this might be due to prediction lens ({in_seq_lens.tolist()}) being smaller than target lens {target_lens.tolist()}\n"
            )

        return ModelOutput(
            out.transpose(0, 1),
            {"ctc_loss": ctc_loss.item()},
            ctc_loss,
        )


class B2tGruExperiment(B2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)
        self.model: B2TModel = self.model

        assert (
            self.config.preprocessing == "seperate_zscoring"
        ), "Only seperate_zscoring is currently supported"

    def get_name(self) -> str:
        return "gru_experiment"

    @staticmethod
    def get_args_model():
        return B2TGruArgsModel

    def _create_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        model = GRUModel(self.config, self.tokenizer)
        return model
