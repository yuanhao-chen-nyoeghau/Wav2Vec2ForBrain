from pydantic import BaseModel
import torch
from src.args.base_args import PRETRAINED_LATENT_SIZES
from src.util.nn_helper import ACTIVATION_FUNCTION
from src.datasets.batch_types import PhonemeSampleBatch
from src.util.nn_helper import create_fully_connected


class BrainFeatureExtractorArgsModel(BaseModel):
    encoder_gru_hidden_size: int = 256
    encoder_bidirectional: bool = True
    encoder_num_gru_layers: int = 2
    encoder_bias: bool = True
    encoder_dropout: float = 0.0
    encoder_learnable_inital_state: bool = False
    encoder_fc_hidden_sizes: list[int] = []
    encoder_fc_activation_function: ACTIVATION_FUNCTION = "gelu"


class BrainFeatureExtractor(torch.nn.Module):
    def __init__(
        self, config: BrainFeatureExtractorArgsModel, in_size, wav2vec_checkpoint: str
    ):
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
            PRETRAINED_LATENT_SIZES[wav2vec_checkpoint],
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
