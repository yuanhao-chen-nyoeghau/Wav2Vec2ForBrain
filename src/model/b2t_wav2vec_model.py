from src.datasets.brain2text import B2tSampleBatch
from src.model.b2tmodel import B2TModel, ModelOutput
from transformers import Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
from src.args.wav2vec_args import (
    B2TWav2VecArgsModel,
    B2TWav2VecCnnArgsModel,
    B2TWav2VecSharedAggregationArgsModel,
)
from torch.nn import Linear, Sequential, ReLU, Flatten, Unflatten, Identity
import torch
from typing import Optional, cast
from src.args.yaml_config import YamlConfigModel
from transformers import PreTrainedTokenizer


class Mean(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=-1)


# brain data is binned in 20 ms bins (50 Hz)
# wav2vec2 was trained on 16kHz audio
# 16000 Hz / 50 Hz = 320
# --> for each 20 ms bin, we need 320 scalars
# naive approach: upsample bins via shared fully connected NN ("shared_fc")
# (Problem: spreads informaton of a single timestamp across the input)


class Brain2AudioShapeModule(torch.nn.Module):
    def __init__(self, config: B2TWav2VecArgsModel):
        super().__init__()
        self.config = cast(
            (
                B2TWav2VecCnnArgsModel
                if config.experiment_type == "b2t_wav2vec_cnn"
                else B2TWav2VecSharedAggregationArgsModel
            ),
            config,
        )
        self.brain2audioshape = self._get_brain2audioshape_module()
        self.activation = self._get_activation()

    def forward(self, batched_input: torch.Tensor) -> torch.Tensor:
        # batched_input shape: (batch_size, timestamps, brain_data)
        audio_shaped_data: torch.Tensor
        if (
            self.config.experiment_type == "b2t_wav2vec_sharedaggregation"
            or self.config.experiment_type == "b2t_wav2vec_pretraining"
        ):
            config = cast(B2TWav2VecSharedAggregationArgsModel, self.config)

            if config.brain2audio_method == "fc":
                result = []
                for t in range(batched_input.size(1)):
                    timestamp_batch = batched_input[:, t, :]
                    result.append(self.brain2audioshape(timestamp_batch))

                # result shape: (timestamps, batch_size, 320)
                # audio shaped data: (batch_size, timestamps * 320)

                # Example (batch_size=3, timestamps=2, 320=2):
                # >>> t1 = torch.tensor([[1,2],[3,4],[5,6]])
                # >>> t2 = torch.tensor([[7,8],[9,10],[11,12]])
                # >>> torch.stack([t1,t2],dim=1)
                # tensor([[[ 1,  2],
                # [ 7,  8]],
                #
                # [[ 3,  4],
                # [ 9, 10]],
                #
                # [[ 5,  6],
                # [11, 12]]])
                audio_shaped_data = torch.stack(result, dim=1).flatten(start_dim=1)
            elif config.brain2audio_method == "mean":
                audio_shaped_data = self.brain2audioshape(batched_input)
            else:
                raise Exception(
                    f"brain2audio_method {config.brain2audio_method} not supported by Brain2AudioShapeModule"
                )
        elif self.config.experiment_type == "b2t_wav2vec_cnn":
            input_with_channel_dim = batched_input.unsqueeze(1)
            # input_with_channel_dim shape: (batch_size, 1, timestamps, brain_data)
            audio_shaped_data = self.brain2audioshape(input_with_channel_dim).flatten(
                start_dim=1
            )
        else:
            raise Exception(
                f"experiment_type {self.config.experiment_type} not supported by Brain2AudioShapeModule"
            )
        return self.activation(audio_shaped_data)

    def _get_activation(self):
        if self.config.activation == "relu":
            return ReLU()
        elif self.config.activation == "identity":
            return Identity()
        raise Exception(f"Unknown activation {self.config.activation}")

    def _get_brain2audioshape_module(self):
        in_size = self._get_timestamp_vec_len()
        if (
            self.config.experiment_type == "b2t_wav2vec_sharedaggregation"
            or self.config.experiment_type == "b2t_wav2vec_pretraining"
        ):
            config = cast(B2TWav2VecSharedAggregationArgsModel, self.config)
            return (
                Sequential(
                    Flatten(), Linear(in_size, config.brain2audio_out_per_sample)
                )
                if config.brain2audio_method == "fc"
                else Mean()
            )
        elif self.config.experiment_type == "b2t_wav2vec_cnn":
            config = cast(B2TWav2VecCnnArgsModel, self.config)

            return torch.nn.Conv2d(
                in_channels=1,
                out_channels=config.brain2audio_cnn_out_channels,
                kernel_size=(config.brain2audio_cnn_bins, in_size),
                stride=(
                    config.brain2audio_cnn_stride_bins,
                    config.brain2audio_cnn_stride_width,
                ),
                padding=(int((config.brain2audio_cnn_bins - 1) / 2), 2),
                dilation=(1, 1),
                groups=1,
                bias=True,
                padding_mode="zeros",
            )
        raise Exception(
            f"experiment_type {self.config.experiment_type} not supported by Brain2AudioShapeModule"
        )

    def _get_timestamp_vec_len(self):
        if self.config.preprocessing == "competition_recommended":
            return 256
        elif self.config.preprocessing == "seperate_zscoring":
            return 256
        elif self.config.preprocessing == "only_tx_unnormalized":
            return 128
        elif self.config.preprocessing == "only_tx_zscored":
            return 128
        elif self.config.preprocessing == "only_spikepow_unnormalized":
            return 128
        elif self.config.preprocessing == "only_spikepow_zscored":
            return 128
        raise Exception(f"Unknown preprocessing type {self.config.preprocessing}")


class B2TWav2Vec(B2TModel):
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
            Wav2Vec2ForCTC,
            Wav2Vec2ForCTC.from_pretrained(
                config.wav2vec_checkpoint,
                cache_dir=yaml_config.cache_dir,
                ctc_loss_reduction=config.ctc_loss_reduction,
                pad_token_id=tokenizer.pad_token_id,
            ),
        )
        print("config", self.wav2vec2.config)
        self.tokenizer = tokenizer

    def forward(self, batch: B2tSampleBatch) -> ModelOutput:
        x, targets = batch
        assert (
            len(x.size()) == 2 or len(x.size()) == 3
        ), "x must be 2D shape: (timestamps, brain_data) or 3D shape: (batch_size, timestamps, brain_data)"

        is_batched = len(x.size()) == 3

        batched_input = x if is_batched else x.unsqueeze(0)
        audio_shaped_data = self.brain2audioshape(batched_input)

        # replace padding token with -100 for it to be ignored in ctc loss
        if targets is not None:
            targets = torch.where(
                targets == self.tokenizer.pad_token_id, torch.tensor(-100), targets
            )

        wav2vec2_out: CausalLMOutput = self.wav2vec2(
            audio_shaped_data, return_dict=True, labels=targets
        )

        metrics = (
            {"ctc_loss": wav2vec2_out.loss.item()}
            if wav2vec2_out.loss is not None
            else {}
        )

        return ModelOutput(
            logits=wav2vec2_out.logits, loss=wav2vec2_out.loss, metrics=metrics
        )
