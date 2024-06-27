from typing import Optional, cast, Tuple
from torch import Tensor, conv2d
from src.datasets.batch_types import B2tSampleBatch
from src.args.b2t_resnet_args import B2TWav2VecResnetArgsModel
from src.model.b2tmodel import B2TModel, ModelOutput
from src.args.yaml_config import YamlConfigModel
from transformers import PreTrainedTokenizer, Wav2Vec2ForCTC, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2ForPreTrainingOutput,
    Wav2Vec2FeatureEncoder,
)
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)
from torch import nn
import torch


class FeatureEncoder(Wav2Vec2FeatureEncoder):
    """Construct the features from raw audio waveform"""

    def __init__(self, config: B2TWav2VecResnetArgsModel):
        super().__init__(config)
        self.gradient_checkpointing = False
        self.config = config

        self.num_out_features = 512

        # in: [4, time, 8, 8]
        self.block1 = self.cnn3d_layer(
            in_features=4,
            out_features=512,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            max_pool_kernel_size=(1, 3, 3),
            max_pool_stride=(1, 2, 2),
            max_pool_padding=(0, 1, 1),
        )
        # in: [64, time, 4, 4]
        self.block2 = self.cnn3d_layer(
            in_features=512,
            out_features=1024,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            max_pool_kernel_size=(1, 3, 3),
            max_pool_stride=(1, 2, 2),
            max_pool_padding=(0, 1, 1),
        )

        # in: [256, time, 2, 2]
        self.block3 = self.cnn3d_layer(
            in_features=1024,
            out_features=2048,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            max_pool_kernel_size=(1, 3, 3),
            max_pool_stride=(1, 2, 2),
            max_pool_padding=(0, 1, 1),
        )
        # out: [512, time, 1, 1]
        if self.config.extractor_head == "linear":
            self.classifier = self.linear_classifier(
                in_features=2048, out_features=self.num_out_features
            )
            # out: [512, time]

        self._requires_grad = True

    def cnn3d_layer(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
        max_pool_kernel_size: Tuple[int, int, int],
        max_pool_stride: Tuple[int, int, int],
        max_pool_padding: Tuple[int, int, int],
    ):
        return nn.Sequential(
            nn.Conv3d(in_features, out_features, kernel_size, stride, padding),
            nn.BatchNorm3d(out_features),
            nn.ReLU(),
            nn.MaxPool3d(max_pool_kernel_size, max_pool_stride, max_pool_padding),
        )

    def linear_classifier(self, in_features: int, out_features: int):
        return nn.Sequential(
            nn.Conv2d(1, out_features, kernel_size=(in_features, 1), stride=(1, 1)),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
        )

    def forward(self, input_values):
        output = self.block1(input_values)
        output = self.block2(output)
        output = self.block3(output)
        output = output.squeeze(-1)
        output = output.squeeze(-1)
        if self.config.extractor_head == "linear":
            output = output.unsqueeze(1)
            output = self.classifier(output)
            output = output.squeeze(2)
        return output


class _CustomEncodeBaseW2VModel(B2TModel):
    def __init__(
        self,
        config: B2TWav2VecResnetArgsModel,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.config = config
        assert (
            self.config.preprocessing == "seperate_zscoring_4channels"
        ), "Preprocessing must be seperate_zscoring_4channels for FeatureEncoder model"
        self.tokenizer = tokenizer

    def _prepare_input(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        assert (
            len(x.size()) == 3 or len(x.size()) == 4
        ), f"x must be 3D shape: (channels=4, timestamps, brain_data=64) or 4D shape: (batch_size, channels=4, timestamps, brain_data=64) but has shape {x.size()}"

        is_batched = len(x.size()) == 4
        batched_input = x if is_batched else x.unsqueeze(0)

        """ batched_input = batched_input.reshape(
            batched_input.size(0), batched_input.size(1), batched_input.size(2), 8, 8
        ) """
        batched_input = batched_input.view(*batched_input.size()[:-1], *(8, 8))
        # replace padding token with -100 for it to be ignored in ctc loss
        if targets is not None:
            targets = torch.where(
                targets == self.tokenizer.pad_token_id, torch.tensor(-100), targets
            )

        return batched_input, targets


class B2TCustomEncoderW2VPretrainingModel(_CustomEncodeBaseW2VModel):
    """Wav2Vec2 model with our own feature encoder, not requiring the data to be converted to audio shape"""

    def __init__(
        self,
        config: B2TWav2VecResnetArgsModel,
        yaml_config: YamlConfigModel,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(config, tokenizer)

        self.wav2vec2 = cast(
            Wav2Vec2ForPreTraining,
            Wav2Vec2ForPreTraining.from_pretrained(
                config.wav2vec_checkpoint,
                cache_dir=yaml_config.cache_dir,
                pad_token_id=tokenizer.pad_token_id,
            ),
        )
        self.wav2vec2.wav2vec2.feature_extractor = FeatureEncoder(config)

    def forward(self, batch: B2tSampleBatch) -> ModelOutput:
        x, targets = batch
        batched_input, targets = self._prepare_input(x, targets)

        batch_size = batched_input.shape[0]
        raw_sequence_length = batched_input.shape[2]

        # TODO: calculate for general case
        sequence_length = raw_sequence_length
        mask_time_indices = _compute_mask_indices(
            shape=(batch_size, int(sequence_length)), mask_prob=0.2, mask_length=2
        )

        sampled_negative_indices = torch.tensor(
            data=_sample_negative_indices(
                features_shape=(batch_size, sequence_length),
                num_negatives=self.wav2vec2.config.num_negatives,
                mask_time_indices=mask_time_indices,
            ),
            device=batched_input.device,
            dtype=torch.long,
        )
        mask_time_indices = torch.tensor(
            data=mask_time_indices, device=batched_input.device, dtype=torch.long
        )

        wav2vec2_out = cast(
            Wav2Vec2ForPreTrainingOutput,
            self.wav2vec2.forward(
                batched_input,
                return_dict=True,
                mask_time_indices=cast(torch.BoolTensor, mask_time_indices),
                sampled_negative_indices=cast(
                    torch.BoolTensor, sampled_negative_indices
                ),
            ),
        )
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


class B2TCustomEncoderW2VFineTuningModel(_CustomEncodeBaseW2VModel):
    """Wav2Vec2 model with our own feature encoder, not requiring the data to be converted to audio shape"""

    def __init__(
        self,
        config: B2TWav2VecResnetArgsModel,
        yaml_config: YamlConfigModel,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(config, tokenizer)
        self.wav2vec2 = cast(
            Wav2Vec2ForCTC,
            Wav2Vec2ForCTC.from_pretrained(
                config.wav2vec_checkpoint,
                cache_dir=yaml_config.cache_dir,
                ctc_loss_reduction=config.ctc_loss_reduction,
                pad_token_id=tokenizer.pad_token_id,
                conv_kernel=config.conv_kernel,
                conv_stride=config.conv_stride,
                num_feat_extract_layers=config.num_feat_extract_layers,
                ignore_mismatched_sizes=True,
            ),
        )
        self.wav2vec2.wav2vec2.feature_extractor = FeatureEncoder(config)

    def forward(self, batch: B2tSampleBatch) -> ModelOutput:
        x, targets = batch
        batched_input, targets = self._prepare_input(x, targets)

        wav2vec2_out = cast(
            CausalLMOutput,
            self.wav2vec2.forward(
                batched_input,
                return_dict=True,
                labels=targets,
                attention_mask=torch.ones(
                    (batched_input.shape[0], batched_input.shape[2]), dtype=torch.long
                ).to(batched_input.device),
            ),
        )
        metrics = (
            {"ctc_loss": wav2vec2_out.loss.item()}
            if wav2vec2_out.loss is not None
            else {}
        )
        return ModelOutput(
            logits=wav2vec2_out.logits, loss=wav2vec2_out.loss, metrics=metrics
        )
