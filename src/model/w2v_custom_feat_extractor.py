from pydantic import BaseModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Config,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2ForCTC,
    Wav2Vec2BaseModelOutput,
    Wav2Vec2Adapter,
    Wav2Vec2Encoder,
)
import torch
from typing import Optional, cast
from src.datasets.batch_types import B2tSampleBatch, PhonemeSampleBatch
from src.model.b2tmodel import B2TModel, ModelOutput
from torch import nn


class W2VBrainEncoderModelArgs(BaseModel):
    w2v_do_stable_layer_norm: bool = False


class W2VBrainEncoderModel(B2TModel):
    def __init__(
        self,
        config: W2VBrainEncoderModelArgs,
        brain_encoder: B2TModel,
        wav2vec_checkpoint: str,
        head: Optional[nn.Module] = None,
        skip_loading_weights: bool = False,
        pre_w2v_head_for_additional_loss: Optional[B2TModel] = None,
        additonal_loss_weight: Optional[float] = None,
        additional_loss_squared: Optional[bool] = False,
    ):
        super().__init__()
        self.brain_encoder = brain_encoder
        w2v_config = cast(
            Wav2Vec2Config,
            Wav2Vec2Config.from_pretrained(
                wav2vec_checkpoint, do_stable_layer_norm=config.w2v_do_stable_layer_norm
            ),
        )
        self.w2v_encoder = cast(
            Wav2Vec2WithoutFeatExtrForCTC,
            (
                (
                    Wav2Vec2WithoutFeatExtrForCTC.from_pretrained(
                        wav2vec_checkpoint,
                        config=w2v_config,
                    )
                )
                if not skip_loading_weights
                else Wav2Vec2WithoutFeatExtrForCTC(w2v_config)
            ),
        )
        if skip_loading_weights:
            print("Skipping loading weights for Wav2Vec2WithoutFeatExtrForCTC\n")
        self.head = head
        self.pre_w2v_head_for_additional_loss = pre_w2v_head_for_additional_loss
        self.loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.additonal_loss_weight = (
            additonal_loss_weight if additonal_loss_weight is not None else 1.0
        )
        self.additional_loss_squared = additional_loss_squared == True

    def forward(self, batch: B2tSampleBatch):
        encoded_brain = self.brain_encoder.forward(batch)
        targets = batch.target
        assert targets is not None

        targets = torch.where(targets < 1, torch.tensor(-100), targets)

        intermediate_out = self._get_intermediate_out(batch, encoded_brain)

        w2v_output, hidden_states = self.w2v_encoder.forward(
            encoded_brain.logits,
        )

        if self.head is not None:
            w2v_output = self.head.forward(w2v_output)

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

        metrics = {}
        if ctc_loss is not None:
            metrics["ctc_loss"] = ctc_loss.item()

        intermediate_loss = None
        if intermediate_out is not None:
            intermediate_loss = intermediate_out.loss
            if intermediate_loss is not None:
                metrics["intermediate_loss"] = intermediate_loss.item()
        loss = (
            ctc_loss
            if intermediate_loss is None
            else ctc_loss
            + (
                (intermediate_loss**2)
                if self.additional_loss_squared
                else intermediate_loss
            )
            * self.additonal_loss_weight
        )
        if intermediate_loss is not None and loss is not None:
            metrics["combined_loss"] = loss.item()
            if self.additional_loss_squared:
                metrics["intermediate_loss_squared"] = (intermediate_loss**2).item()
        return ModelOutput(
            w2v_output,
            metrics,
            loss=loss,
            logit_lens=encoded_brain.logit_lens,
            hidden_states=hidden_states,
        )

    def _get_intermediate_out(self, batch: B2tSampleBatch, encoded_brain: ModelOutput):
        if self.pre_w2v_head_for_additional_loss is None:
            return None
        intermediate_batch = PhonemeSampleBatch(encoded_brain.logits, batch.target)
        assert (
            encoded_brain.logit_lens is not None
        ), "logit_lens must be defined if pre_w2v_head_for_additional_loss is not None"
        intermediate_batch.input_lens = encoded_brain.logit_lens
        intermediate_batch.target_lens = batch.target_lens
        intermediate_out = self.pre_w2v_head_for_additional_loss.forward(
            intermediate_batch
        )
        return intermediate_out


class Wav2Vec2WithoutFeatExtrForCTC(Wav2Vec2ForCTC):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config, target_lang)
        self.wav2vec2 = Wav2Vec2WithoutFeatExtrModel(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(
            x,
            return_dict=True,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)
        return logits, hidden_states


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
