from pydantic import BaseModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Config,
)
import torch
from torch import nn
from typing import Literal, cast

from src.model.w2v_no_encoder import Wav2Vec2WithoutTransformerModel
from src.model.suc import SUCModel
from src.datasets.timit_dataset import TimitSampleBatch
from src.args.wav2vec_args import ACTIVATION_FUNCTION
from src.model.b2tmodel import B2TModel, ModelOutput


class W2VSUCArgsModel(BaseModel):
    suc_hidden_sizes: list[int] = []
    suc_hidden_activation: ACTIVATION_FUNCTION = "gelu"
    suc_dropout: float = 0.0
    disable_w2v_feature_extractor_grad: bool = True


class W2VSUCModel(B2TModel):
    def __init__(self, config: W2VSUCArgsModel, class_weights: torch.Tensor):
        super().__init__()
        self.config = config
        w2v_config = cast(
            Wav2Vec2Config,
            Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h"),
        )
        self.w2v_feature_extractor = cast(
            Wav2Vec2WithoutTransformerModel,
            Wav2Vec2WithoutTransformerModel.from_pretrained(
                "facebook/wav2vec2-base-960h", config=w2v_config
            ),
        )
        self.dropout = nn.Dropout(config.suc_dropout)
        self.suc = SUCModel(
            hidden_sizes=config.suc_hidden_sizes,
            activation=config.suc_hidden_activation,
        )

        self.loss = nn.CrossEntropyLoss(reduction="mean", weight=class_weights.cuda())

    def forward(self, batch: TimitSampleBatch) -> ModelOutput:
        if batch.target is None:
            raise ValueError("Target is required for training")

        with torch.no_grad():
            w2v_output = self.w2v_feature_extractor(batch.input)
        w2v_output = self.dropout(w2v_output)
        suc_output = self.suc(w2v_output).squeeze(1)

        loss = self.loss.forward(suc_output, batch.target)
        metrics = {"cle": loss.item()}

        return ModelOutput(
            suc_output,
            metrics,
            loss=loss,
        )
