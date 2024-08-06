from typing import Optional
from pydantic import BaseModel
import torch
from torch import nn
from src.args.base_args import PRETRAINED_LATENT_SIZES
from src.datasets.batch_types import SampleBatch
from src.util.nn_helper import ACTIVATION_FUNCTION
from src.model.b2tmodel import B2TModel, ModelOutput
from src.util.nn_helper import create_fully_connected


class DiscriminatorModelArgsModel(BaseModel):
    discriminator_hidden_sizes: list[int] = []
    discriminator_hidden_activation: ACTIVATION_FUNCTION = "gelu"
    discriminator_dropout: float = 0.0


class DiscriminatorModel(B2TModel):
    def __init__(
        self,
        config: DiscriminatorModelArgsModel,
        w2v_to_brain_samples_ratio: Optional[float],
        wav2vec_checkpoint: str,
    ):
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(config.discriminator_dropout)
        self.discriminator = create_fully_connected(
            PRETRAINED_LATENT_SIZES[wav2vec_checkpoint],
            1,
            config.discriminator_hidden_sizes,
            config.discriminator_hidden_activation,
        )

        self.loss = (
            nn.BCEWithLogitsLoss(
                reduction="mean", pos_weight=torch.tensor([w2v_to_brain_samples_ratio])
            )
            if w2v_to_brain_samples_ratio is not None
            else None
        )

    def forward(self, batch: SampleBatch) -> ModelOutput:
        out = self.dropout(batch.input)
        out = self.discriminator(out)

        if self.loss is None or batch.target is None:
            return ModelOutput(torch.sigmoid(out), {})

        loss = self.loss.forward(out, batch.target)
        metrics = {"bce": loss.item()}

        return ModelOutput(
            torch.sigmoid(out),
            metrics,
            loss=loss,
        )
