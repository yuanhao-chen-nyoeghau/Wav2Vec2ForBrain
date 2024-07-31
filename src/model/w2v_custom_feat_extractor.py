from pydantic import BaseModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Config,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2ForCTC,
    CausalLMOutput,
    Wav2Vec2BaseModelOutput,
    Wav2Vec2Adapter,
    Wav2Vec2Encoder,
)
import torch
from typing import Optional, cast
from src.datasets.batch_types import PhonemeSampleBatch
from src.model.b2tmodel import B2TModel, ModelOutput
from src.model.b2p2t_model import B2P2TModel


class W2VBrainEncoderModelArgs(BaseModel):
    pass


class W2VBrainEncoderModel(B2TModel):
    def __init__(self, config: W2VBrainEncoderModelArgs, brain_encoder: B2P2TModel):
        self.brain_encoder = brain_encoder
        w2v_config = cast(
            Wav2Vec2Config,
            Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h"),
        )
        self.w2v_encoder = cast(
            Wav2Vec2WithoutFeatExtrForCTC,
            Wav2Vec2WithoutFeatExtrForCTC.from_pretrained(
                "facebook/wav2vec2-base-960h", config=w2v_config
            ),
        )

    def forward(self, batch: PhonemeSampleBatch):
        encoded_brain = self.brain_encoder.forward(batch)
        targets = batch.target
        assert targets is not None
        targets = torch.where(targets < 1, torch.tensor(-100), targets)
        w2v_output = cast(
            CausalLMOutput,
            self.w2v_encoder.forward(encoded_brain.logits, labels=targets),
        )
        loss = w2v_output.loss
        assert loss is not None

        return ModelOutput(w2v_output.logits, {"ctc_loss": loss.item()}, loss=loss)


class Wav2Vec2WithoutFeatExtrForCTC(Wav2Vec2ForCTC):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config, target_lang)
        self.wav2vec2 = Wav2Vec2WithoutFeatExtrModel(config)


class Wav2Vec2WithoutFeatExtrModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)
        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values: torch.FloatTensor,
    ) -> Wav2Vec2BaseModelOutput:
        encoder_outputs = self.encoder(
            input_values,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = encoder_outputs[0]
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=input_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
