from typing import Optional
from pydantic import BaseModel
import torch
from src.datasets.batch_types import SampleBatch
from src.model.b2p2t_model import B2P2TModel, B2P2TModelArgsModel
from src.model.b2tmodel import B2TModel, ModelOutput
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


class B2TBrainFeatureExtractor(B2TModel):
    def __init__(
        self,
        config: BrainFeatureExtractorArgsModel,
        wav2vec_checkpoint: str,
        in_size: int,
    ):
        super().__init__()
        self.encoder = BrainFeatureExtractor(
            config,
            in_size,
            wav2vec_checkpoint,
        )

    def forward(self, batch: SampleBatch) -> ModelOutput:
        out = self.encoder(batch)
        return ModelOutput(logits=out, metrics={})


class B2P2TBrainFeatureExtractorArgsModel(
    BrainFeatureExtractorArgsModel, B2P2TModelArgsModel
):
    pass


def bfe_w_preprocessing_from_config(
    config: B2P2TBrainFeatureExtractorArgsModel,
    brain_encoder_path: Optional[str],
    wav2vec_checkpoint: str,
):
    brain_feat_extractor = B2P2TModel(
        config,
        B2TBrainFeatureExtractor(
            config,
            wav2vec_checkpoint,
            B2P2TModel.get_in_size_after_preprocessing(config.unfolder_kernel_len),
        ),
    ).cuda()
    if brain_encoder_path != None:
        state = torch.load(brain_encoder_path, map_location="cuda")
        unneeded_keys = [
            key
            for key in state.keys()
            if key.startswith("neural_decoder.discriminator")
            or key.startswith("neural_decoder.suc_for_ctc")
        ]
        for key in unneeded_keys:
            del state[key]
        brain_feat_extractor.load_state_dict(
            state,
            strict=True,
        )
    return brain_feat_extractor
