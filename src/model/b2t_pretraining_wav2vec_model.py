from src.datasets.brain2text import B2tSampleBatch
from src.model.b2tmodel import B2TModel, ModelOutput
from transformers import Wav2Vec2ForPreTraining
from transformers.modeling_outputs import CausalLMOutput
from src.args.wav2vec_args import B2TWav2VecArgsModel
from src.model.b2t_wav2vec_model import Brain2AudioShapeModule
import torch
from typing import Optional, cast
from src.args.yaml_config import YamlConfigModel
from transformers import PreTrainedTokenizer
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTrainingOutput


class B2TPretrainingWav2VecModel(B2TModel):
    def __init__(
        self,
        config: B2TWav2VecArgsModel,
        yaml_config: YamlConfigModel,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.config = config

        self.brain2audioshape = Brain2AudioShapeModule(config)
        self.wav2vec2 = cast(
            Wav2Vec2ForPreTraining,
            Wav2Vec2ForPreTraining.from_pretrained(
                config.wav2vec_checkpoint,
                cache_dir=yaml_config.cache_dir,
                pad_token_id=tokenizer.pad_token_id,
            ),
        )
        self.tokenizer = tokenizer

    def forward(self, batch: B2tSampleBatch) -> ModelOutput:
        x, targets = batch
        assert (
            len(x.size()) == 2 or len(x.size()) == 3
        ), "x must be 2D shape: (timestamps, brain_data) or 3D shape: (batch_size, timestamps, brain_data)"

        is_batched = len(x.size()) == 3

        batched_input = x if is_batched else x.unsqueeze(0)
        audio_shaped_data = self.brain2audioshape(batched_input)

        batch_size, raw_sequence_length = audio_shaped_data.shape
        sequence_length = cast(
            torch.LongTensor,
            self.wav2vec2._get_feat_extract_output_lengths(raw_sequence_length),
        ).item()
        mask_time_indices = _compute_mask_indices(
            shape=(batch_size, int(sequence_length)), mask_prob=0.2, mask_length=2
        )

        sampled_negative_indices = (
            torch.tensor(
                data=_sample_negative_indices(
                    features_shape=(batch_size, sequence_length),
                    num_negatives=self.wav2vec2.config.num_negatives,
                    mask_time_indices=mask_time_indices,
                ),
                device=audio_shaped_data.device,
                dtype=torch.long,
            )
            if self.training
            else None
        )
        mask_time_indices = torch.tensor(
            data=mask_time_indices, device=audio_shaped_data.device, dtype=torch.long
        )
        wav2vec2_out = cast(
            Wav2Vec2ForPreTrainingOutput,
            self.wav2vec2.forward(
                audio_shaped_data,
                mask_time_indices=cast(torch.BoolTensor, mask_time_indices),
                sampled_negative_indices=cast(
                    torch.BoolTensor, sampled_negative_indices
                ),
            ),
        )
        # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
        cosine_sim = torch.cosine_similarity(
            wav2vec2_out.projected_states,
            wav2vec2_out.projected_quantized_states,
            dim=-1,
        )

        metrics = {"cosine_similarity": cosine_sim.mean().item()}
        if wav2vec2_out.loss is not None:
            metrics["contrastive_loss"] = wav2vec2_out.loss.item()

        return ModelOutput(
            logits=torch.tensor([]), loss=wav2vec2_out.loss, metrics=metrics
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.wav2vec2.train(mode)
