from src.args.base_args import BaseExperimentArgsModel, B2TDatasetArgsModel
from typing import Literal, Optional


class B2TWav2VecArgsModel(BaseExperimentArgsModel, B2TDatasetArgsModel):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h", "facebook/wav2vec2-base-960h"
    ] = "facebook/wav2vec2-base-960h"
    unfreeze_strategy: Literal[
        "wav2vec2featureextractor_ours", "all"
    ] = "wav2vec2featureextractor_ours"
    remove_punctuation: bool = True
    tokenizer: Literal["wav2vec_pretrained", "ours"] = "wav2vec_pretrained"
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


class B2TAudioWav2VecArgsModel(AudioWav2VecArgsModel):
    hidden_nodes: int = 16
    mean_reduction: bool = False
