from src.experiments.b2t_experiment import B2TArgsModel
from src.args.base_args import BaseExperimentArgsModel
from typing import Literal, Optional
from transformers.activations import ACT2FN


class B2TWav2VecArgsModel(B2TArgsModel):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h",
        "facebook/wav2vec2-base-960h",
        "facebook/wav2vec2-xls-r-300m",
    ] = "facebook/wav2vec2-base-960h"
    unfreeze_strategy: Literal[
        "wav2vec2featureextractor_ours", "all", "ours", "lm_head"
    ] = "wav2vec2featureextractor_ours"
    activation: Literal["identity", "relu"] = "identity"


class B2TWav2VecSharedAggregationArgsModel(B2TWav2VecArgsModel):
    brain2audio_method: Literal["fc", "mean"] = "mean"
    brain2audio_out_per_sample: int = 320


class B2TWav2VecCnnArgsModel(B2TWav2VecArgsModel):
    brain2audio_cnn_bins: int = 3
    brain2audio_cnn_kernel_width: int = 128
    brain2audio_cnn_stride_width: int = 1
    brain2audio_cnn_stride_bins: int = 1
    brain2audio_cnn_out_channels: int = 320


class AudioWav2VecArgsModel(BaseExperimentArgsModel):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h", "facebook/wav2vec2-base-960h"
    ] = "facebook/wav2vec2-base-960h"
    unfreeze_strategy: Literal["wav2vec2featureextractor", "all"] = "all"
    tokenizer: Literal["wav2vec_pretrained"] = "wav2vec_pretrained"
    remove_punctuation: bool = True


ACTIVATION_FUNCTION = Literal[
    "gelu",
    "gelu_10",
    "gelu_fast",
    "gelu_new",
    "gelu_python",
    "gelu_pytorch_tanh",
    "gelu_accurate",
    "laplace",
    "linear",
    "mish",
    "quick_gelu",
    "relu",
    "relu2",
    "relu6",
    "sigmoid",
    "silu",
    "swish",
    "tanh",
]


class B2TWav2VecCustomEncoderArgsModel(B2TWav2VecArgsModel):
    mode: Literal["pretraining", "finetuning"] = "pretraining"
    conv_bias: bool = True
    conv_stride: list[int] = [1, 1, 1, 1]  # [5, 2, 2, 2, 2, 2, 2]
    conv_kernel: list[int] = [5, 3, 3, 3]  # [10, 3, 3, 3, 3, 2, 2]
    conv_dim: list[int] = [512, 512, 512, 512]  # [512, 512, 512, 512, 512, 512, 512]
    feat_extract_activation: ACTIVATION_FUNCTION = "gelu"
    feat_extract_norm: Literal["group", "layer"] = "group"
    num_feat_extract_layers: int = 4  # 7
