from typing import Optional
from pydantic import BaseModel
import torch

from src.args.base_args import PRETRAINED_LATENT_SIZES
from src.model.discriminator_model import (
    DiscriminatorModel,
    DiscriminatorModelArgsModel,
)
from src.util.nn_helper import ACTIVATION_FUNCTION
from src.datasets.batch_types import PhonemeSampleBatch, SampleBatch

from src.model.b2tmodel import B2TModel, ModelOutput
from src.model.w2v_suc_ctc_model import SUCForCTC, W2VSUC_CTCArgsModel
from src.util.nn_helper import create_fully_connected


class BrainEncoderArgsModel(BaseModel):
    encoder_gru_hidden_size: int = 256
    encoder_bidirectional: bool = True
    encoder_num_gru_layers: int = 2
    encoder_bias: bool = True
    encoder_dropout: float = 0.0
    encoder_learnable_inital_state: bool = False
    encoder_fc_hidden_sizes: list[int] = []
    encoder_fc_activation_function: ACTIVATION_FUNCTION = "gelu"


class B2PSUCArgsModel(
    W2VSUC_CTCArgsModel, BrainEncoderArgsModel, DiscriminatorModelArgsModel
):
    discriminator_checkpoint: Optional[str] = None
    discriminator_loss_weight: float = 1.0


class BrainEncoder(torch.nn.Module):
    def __init__(self, config: BrainEncoderArgsModel, in_size, wav2vec_checkpoint: str):
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


class B2PSUC(B2TModel):
    def __init__(self, config: B2PSUCArgsModel, in_size: int, wav2vec_checkpoint: str):
        super().__init__()
        self.config = config
        self.suc_for_ctc = SUCForCTC(config)
        self.encoder = BrainEncoder(config, in_size, wav2vec_checkpoint)
        self.ctc_loss: torch.nn.CTCLoss = torch.nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=True
        )
        if config.discriminator_checkpoint is not None:
            self.discriminator = DiscriminatorModel(config, None, wav2vec_checkpoint)

    def forward(self, batch: PhonemeSampleBatch) -> ModelOutput:
        encoder_out = self.encoder.forward(batch)

        suc_out = self.suc_for_ctc.forward(encoder_out)

        ctc_loss = (
            self.ctc_loss.forward(
                torch.permute(torch.log_softmax(suc_out, -1), [1, 0, 2]),
                batch.target,
                batch.input_lens,
                batch.target_lens,
            )
            if batch.target_lens is not None and batch.target is not None
            else None
        )
        discriminator_loss = (
            self.discriminator.forward(SampleBatch(encoder_out, None)).logits.mean()
            if hasattr(self, "discriminator")
            else None
        )

        metrics_dict = {"ctc_loss": ctc_loss.item()} if ctc_loss is not None else {}

        loss = (
            ctc_loss + self.config.discriminator_loss_weight * discriminator_loss
            if ctc_loss is not None and discriminator_loss is not None
            else ctc_loss
        )

        if discriminator_loss is not None:
            metrics_dict["discriminator_loss"] = discriminator_loss.item()
            if loss is not None:
                metrics_dict["combined_loss"] = loss.item()

        return ModelOutput(logits=suc_out, metrics=metrics_dict, loss=loss)
