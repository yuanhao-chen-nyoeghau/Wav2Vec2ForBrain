import torch

from src.args.wav2vec_args import ACTIVATION_FUNCTION
from src.datasets.batch_types import PhonemeSampleBatch

from src.model.b2tmodel import B2TModel, ModelOutput
from src.model.w2v_suc_seq_model import SUCForCTC, W2VSUC_CTCArgsModel
from src.util.nn_helper import create_fully_connected


class B2PSUCArgsModel(W2VSUC_CTCArgsModel):
    encoder_gru_hidden_size: int = 256
    encoder_bidirectional: bool = True
    encoder_num_gru_layers: int = 2
    encoder_bias: bool = True
    encoder_dropout: float = 0.0
    encoder_learnable_inital_state: bool = False
    encoder_fc_hidden_sizes: list[int] = []
    encoder_fc_activation_function: ACTIVATION_FUNCTION = "gelu"


class BrainEncoder(torch.nn.Module):
    def __init__(self, config: B2PSUCArgsModel, in_size):
        super().__init__()
        self.config = config
        self.num_directions = 2 if config.encoder_bidirectional else 1
        self.hidden_start = torch.nn.Parameter(
            torch.randn(
                self.num_directions * config.encoder_num_gru_layers,
                config.encoder_gru_hidden_size,
                requires_grad=True,
            )
        )

        self.gru = torch.nn.GRU(
            in_size,
            config.encoder_gru_hidden_size,
            config.encoder_num_gru_layers,
            dropout=config.encoder_dropout,
            bias=config.encoder_bias,
            bidirectional=config.encoder_bidirectional,
            batch_first=True,
        )

        self.fc = create_fully_connected(
            config.encoder_gru_hidden_size * self.num_directions,
            768,
            config.encoder_fc_hidden_sizes,
            config.encoder_fc_activation_function,
        )

    def forward(self, batch: PhonemeSampleBatch) -> torch.Tensor:
        x, _ = batch

        batch_size = x.shape[0]

        out, _ = (
            self.gru(x, self.hidden_start.unsqueeze(1).repeat(1, batch_size, 1))
            if self.config.encoder_learnable_inital_state
            else self.gru(x)
        )

        out = self.fc(out)
        return out


class B2PSUC(B2TModel):
    def __init__(self, config: B2PSUCArgsModel, in_size: int):
        super().__init__()
        self.config = config
        self.suc_for_ctc = SUCForCTC(config)
        self.encoder = BrainEncoder(config, in_size)
        self.loss: torch.nn.CTCLoss = torch.nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=True
        )

    def forward(self, batch: PhonemeSampleBatch) -> ModelOutput:
        out = self.encoder.forward(batch)

        out = self.suc_for_ctc.forward(out)

        ctc_loss = (
            self.loss.forward(
                torch.permute(torch.log_softmax(out, -1), [1, 0, 2]),
                batch.target,
                batch.input_lens,
                batch.target_lens,
            )
            if batch.target_lens is not None and batch.target is not None
            else None
        )

        metrics_dict = {"ctc_loss": ctc_loss.item()} if ctc_loss is not None else {}

        return ModelOutput(logits=out, metrics=metrics_dict, loss=ctc_loss)
