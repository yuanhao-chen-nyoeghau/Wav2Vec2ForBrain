from typing import Optional, Tuple, Union, cast
import torch
from torch import FloatTensor, Tensor
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model
from transformers.modeling_outputs import CausalLMOutput, Wav2Vec2BaseModelOutput
from torch.nn import Module
from src.args.b2t_audio_args import B2TAudioWav2VecArgsModel
from src.args.yaml_config import YamlConfigModel
from torch import nn

_HIDDEN_STATES_START_POSITION = 2


class MultiWaveInputWav2VecCTC(Wav2Vec2ForCTC):
    def __init__(self, wav2vec2forCTC: Wav2Vec2ForCTC):
        super().__init__(wav2vec2forCTC.config)

        # Putting in pretrained components
        self.wav2vec2 = MultiChannelWav2Vec2Model(wav2vec2forCTC.wav2vec2)
        self.dropout = wav2vec2forCTC.dropout

        self.target_lang = wav2vec2forCTC.target_lang

        self.lm_head = wav2vec2forCTC.lm_head

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(
                    f"Label values must be <= vocab_size: {self.config.vocab_size}"
                )

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )

            # Fixes input lengths for higher dimensional input
            if len(attention_mask.size()) == 3:
                input_lengths = self._get_feat_extract_output_lengths(
                    torch.tensor([attention_mask.size(1)] * attention_mask.size(0)).to(
                        torch.long
                    )
                ).to(torch.long)
            else:
                input_lengths = self._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)
                ).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MultiChannelWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, wav2vec2_model: Wav2Vec2Model):
        super().__init__(wav2vec2_model.config)

        # Put in pretrained models
        self.feature_extractor = wav2vec2_model.feature_extractor
        self.feature_projection = wav2vec2_model.feature_projection
        self.encoder = wav2vec2_model.encoder
        self.adapter = wav2vec2_model.adapter
        self.summarizer = torch.nn.Linear(64, 1)

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
        mask_time_indices: FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> Tuple | Wav2Vec2BaseModelOutput:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        channels = input_values.size(-1)

        # Added sequential feature extraction for every soundwave
        features_per_channel = []

        for i in range(channels):
            audio_wave = input_values[:, :, i]
            extract_features = self.feature_extractor(audio_wave).transpose(1, 2)
            features_per_channel.append(extract_features)

        extract_features = torch.stack(features_per_channel, 1)
        extract_features = torch.mean(extract_features, 1).squeeze(1)

        hidden_states, extract_features = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(
            hidden_states,
            mask_time_indices=mask_time_indices,
            attention_mask=attention_mask,
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
