from pydantic import BaseModel
from src.datasets.brain2text import B2tSampleBatch
from src.args.wav2vec_args import ACTIVATION_FUNCTION
from src.model.b2tmodel import B2TModel, ModelOutput
import torch
from torch import nn
from torch.nn.functional import log_softmax
from src.util.nn_helper import create_fully_connected


class GruArgsModel(BaseModel):
    hidden_size: int = 256
    bidirectional: bool = True
    num_gru_layers: int = 2
    bias: bool = True
    dropout: float = 0.0
    learnable_inital_state: bool = False
    classifier_hidden_sizes: list[int] = []
    classifier_activation: ACTIVATION_FUNCTION = "gelu"


# Source: https://github.com/cffan/neural_seq_decoder/blob/master/src/neural_decoder/augmentations.py#L27


class GRUModel(B2TModel):
    def __init__(
        self, config: GruArgsModel, vocab_size: int, in_size: int, pad_token_id=0
    ):
        super().__init__()
        self.config = config
        self.num_directions = 2 if config.bidirectional else 1
        self.pad_token_id = pad_token_id
        self.hidden_start = nn.Parameter(
            torch.randn(
                self.num_directions * config.num_gru_layers,
                config.hidden_size,
                requires_grad=True,
            )
        )

        self.gru = torch.nn.GRU(
            in_size,
            config.hidden_size,
            config.num_gru_layers,
            dropout=config.dropout,
            bias=config.bias,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        self.classifier = create_fully_connected(
            config.hidden_size * self.num_directions,
            vocab_size,
            config.classifier_hidden_sizes,
            config.classifier_activation,
        )

        self.loss: nn.CTCLoss = nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=True
        )

    def forward(self, batch: B2tSampleBatch) -> ModelOutput:
        x, targets = batch

        assert targets is not None, "Targets must be set"
        if targets is not None:
            targets = torch.where(
                targets == self.pad_token_id, torch.tensor(-100), targets
            )

        batch_size = x.shape[0]

        out, h_out = (
            self.gru(x, self.hidden_start.unsqueeze(1).repeat(1, batch_size, 1))
            if self.config.learnable_inital_state
            else self.gru(x)
        )

        # out shape: (batch_size, seq_len, hidden_size * num_directions)
        out = self.classifier(out)

        # out shape: (batch_size, seq_len, vocab_size)

        out = log_softmax(out, -1)
        # TODO: check if ctc loss calculation is still valid
        ctc_loss = self.loss.forward(
            torch.permute(out, [1, 0, 2]),
            targets,
            batch.input_lens,
            batch.target_lens,
        )

        return ModelOutput(
            out,
            {"ctc_loss": ctc_loss.item()},
            ctc_loss,
        )
