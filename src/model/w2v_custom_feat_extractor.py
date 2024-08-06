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
from src.datasets.batch_types import B2tSampleBatch, PhonemeSampleBatch
from src.model.b2tmodel import B2TModel, ModelOutput
from src.model.b2p2t_model import B2P2TModel


class W2VBrainEncoderModelArgs(BaseModel):
    pass


class W2VBrainEncoderModel(B2TModel):
    def __init__(
        self,
        config: W2VBrainEncoderModelArgs,
        brain_encoder: B2TModel,
        wav2vec_checkpoint: str,
    ):
        super().__init__()
        self.brain_encoder = brain_encoder
        w2v_config = cast(
            Wav2Vec2Config,
            Wav2Vec2Config.from_pretrained(wav2vec_checkpoint),
        )
        self.w2v_encoder = cast(
            Wav2Vec2WithoutFeatExtrForCTC,
            Wav2Vec2WithoutFeatExtrForCTC.from_pretrained(
                wav2vec_checkpoint, config=w2v_config
            ),
        )
        self.loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(self, batch: B2tSampleBatch):
        encoded_brain = self.brain_encoder.forward(batch)
        targets = batch.target
        assert targets is not None

        targets = torch.where(targets < 1, torch.tensor(-100), targets)
        w2v_output = self.w2v_encoder.forward(encoded_brain.logits)

        ctc_loss = (
            self.loss.forward(
                torch.log_softmax(w2v_output, -1).transpose(0, 1),
                targets,
                encoded_brain.logit_lens.cuda(),
                batch.target_lens.cuda(),
            )
            if batch.target_lens is not None and encoded_brain.logit_lens is not None
            else None
        )

        return ModelOutput(
            w2v_output,
            {"ctc_loss": ctc_loss.item()} if ctc_loss is not None else {},
            loss=ctc_loss,
        )


class Wav2Vec2WithoutFeatExtrForCTC(Wav2Vec2ForCTC):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config, target_lang)
        self.wav2vec2 = Wav2Vec2WithoutFeatExtrModel(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.wav2vec2(
            x,
            return_dict=True,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)
        return logits


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

    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
