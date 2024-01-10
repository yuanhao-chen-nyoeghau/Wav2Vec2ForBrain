from typing import Optional, cast
from torch import Tensor, conv2d
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
from torchvision.models import resnet152
from torch import nn
import torch


IN_CHANNELS = 2
NUM_FEATURES = 128


class FeatureEncoder(Wav2Vec2FeatureEncoder):
    """Construct the features from raw audio waveform"""

    def __init__(self, config: B2TWav2VecResnetArgsModel):
        super().__init__(config)
        self.resnet = resnet152(pretrained=True)
        self.gradient_checkpointing = False
        self.conv_transformer = nn.Conv3d(
            4, 3, (1, 3, 3), stride=1, padding=(0, 1, 1)
        ).cuda()
        num_in_features = self.resnet.fc.in_features
        self.num_out_features = 512
        self.resnet.fc = nn.Linear(num_in_features, self.num_out_features)
        self.resnet = self.resnet.cuda()
        self._requires_grad = True

    def forward(self, input_values):
        hidden_states = input_values
        hidden_states = self.conv_transformer(hidden_states)

        batch_length = hidden_states.size(0)
        input_length = hidden_states.size(2)

        hidden_states = hidden_states.view(
            -1, hidden_states.size(1), hidden_states.size(3), hidden_states.size(4)
        )
        hidden_states = nn.functional.interpolate(
            hidden_states, size=(64, 64), mode="bilinear"
        )
        resnet_output = self.resnet(hidden_states)
        resnet_output = resnet_output.view(
            batch_length, input_length, resnet_output.size(1)
        ).permute(0, 2, 1)

        """ resnet_output = torch.zeros(
            [batch_length, self.num_out_features, input_length]
        ).cuda()
        for i in range(hidden_states.size(2)):
            input_image = hidden_states[:, :, i, :, :].squeeze(2)
            input_image = nn.functional.interpolate(
                input_image, size=(64, 64), mode="bilinear"
            )
            features = self.resnet(input_image)
            resnet_output[:, :, i] = features """

        return resnet_output


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
        ), "Preprocessing must be seperate_zscoring_2channels for CustomFeatureEncoder model"
        self.tokenizer = tokenizer

    def _prepare_input(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        assert (
            len(x.size()) == 3 or len(x.size()) == 4
        ), f"x must be 3D shape: (channels=4, timestamps, brain_data=64) or 4D shape: (batch_size, channels=4, timestamps, brain_data=64) but has shape {x.size()}"

        is_batched = len(x.size()) == 4
        batched_input = x if is_batched else x.unsqueeze(0)
        batched_input = batched_input.view(
            batched_input.size(0), batched_input.size(1), batched_input.size(2), 8, 8
        )
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

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> ModelOutput:
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
        metrics = (
            {"contrastive_loss": wav2vec2_out.loss.item()}
            if wav2vec2_out.loss is not None
            else {}
        )
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
            ),
        )
        self.wav2vec2.wav2vec2.feature_extractor = FeatureEncoder(config)

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        batched_input, targets = self._prepare_input(x, targets)

        wav2vec2_out = cast(
            CausalLMOutput,
            self.wav2vec2.forward(batched_input, return_dict=True),
        )
        metrics = (
            {"ctc_loss": wav2vec2_out.loss.item()}
            if wav2vec2_out.loss is not None
            else {}
        )
        return ModelOutput(
            logits=wav2vec2_out.logits, loss=wav2vec2_out.loss, metrics=metrics
        )
