from pydantic import BaseModel
import torch
from torch import nn
from src.datasets.batch_types import SampleBatch
from src.args.wav2vec_args import ACTIVATION_FUNCTION
from src.model.b2tmodel import B2TModel, ModelOutput
from src.util.nn_helper import create_fully_connected


class DiscriminatorModelArgsModel(BaseModel):
    discriminator_hidden_sizes: list[int] = []
    discriminator_hidden_activation: ACTIVATION_FUNCTION = "gelu"
    discriminator_dropout: float = 0.0


class DiscriminatorModel(B2TModel):
    def __init__(
        self, config: DiscriminatorModelArgsModel, w2v_to_brain_samples_ratio: float
    ):
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(config.discriminator_dropout)
        self.discriminator = create_fully_connected(
            768,
            1,
            config.discriminator_hidden_sizes,
            config.discriminator_hidden_activation,
        )

        self.loss = nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=torch.tensor([w2v_to_brain_samples_ratio])
        )

    def forward(self, batch: SampleBatch) -> ModelOutput:

        if batch.target is None:
            raise ValueError("Target is required for training")

        out = self.dropout(batch.input)
        out = self.discriminator(out)

        loss = self.loss.forward(out, batch.target)
        metrics = {"bce": loss.item()}

        return ModelOutput(
            torch.sigmoid(out),
            metrics,
            loss=loss,
        )
