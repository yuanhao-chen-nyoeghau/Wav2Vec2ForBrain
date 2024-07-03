from pydantic import BaseModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Config,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2BaseModelOutput,
)
import torch
from torch import log_softmax, nn
from typing import Optional, Tuple, Union, cast

from src.args.wav2vec_args import ACTIVATION_FUNCTION
from src.datasets.batch_types import PhonemeSampleBatch
from src.datasets.brain2text_w_phonemes import PHONE_DEF_SIL
from src.model.b2tmodel import B2TModel, ModelOutput
from src.util.nn_helper import create_fully_connected


class W2VSUCArgsModel(BaseModel):
    suc_hidden_sizes: list[int] = []
    suc_hidden_activation: ACTIVATION_FUNCTION = "gelu"


class W2VSUCModel(B2TModel):
    def __init__(self, config: W2VSUCArgsModel):
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
        self.suc = create_fully_connected(
            768, len(PHONE_DEF_SIL) + 1, config.suc_hidden_sizes
        )
        self.loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(self, batch: PhonemeSampleBatch) -> ModelOutput:
        if batch.target is None or batch.target_lens is None:
            raise ValueError("Target and target_lens are required for training")

        device = batch.input.device
        w2v_output = self.w2v_feature_extractor(batch.input)
        suc_output = self.suc(w2v_output)

        feature_extract_output_lens = cast(
            torch.LongTensor,
            self.w2v_feature_extractor._get_feat_extract_output_lengths(
                cast(torch.LongTensor, batch.input_lens.long())
            ),
        )

        ctc_loss = self.loss.forward(
            log_softmax(suc_output, -1).transpose(0, 1),
            batch.target,
            feature_extract_output_lens.to(device),
            batch.target_lens.to(device),
        )
        return ModelOutput(
            suc_output,
            {"ctc_loss": ctc_loss.item()},
            loss=ctc_loss,
            logit_lens=feature_extract_output_lens,
        )


class Wav2Vec2WithoutTransformerModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)
        # TODO: load pretrained weights
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        hidden_states, extract_features = self.feature_projection(extract_features)
        return hidden_states
