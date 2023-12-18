from src.args.base_args import BaseExperimentArgsModel, B2TDatasetArgsModel
from typing import Literal, Optional


class B2TWav2VecArgsModel(BaseExperimentArgsModel, B2TDatasetArgsModel):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h", "facebook/wav2vec2-base-960h"
    ] = "facebook/wav2vec2-base-960h"
    brain2audioshape_strategy: Literal[
        "shared_fc", "shared_fc_relu", "2d_cnn"
    ] = "shared_fc"
    unfreeze_strategy: Literal[
        "wav2vec2featureextractor_ours", "all"
    ] = "wav2vec2featureextractor_ours"
    brain2audio_cnn_bins: Optional[int] = None


class AudioWav2VecArgsModel(BaseExperimentArgsModel):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h", "facebook/wav2vec2-base-960h"
    ] = "facebook/wav2vec2-base-960h"
    unfreeze_strategy: Literal["wav2vec2featureextractor", "all"] = "all"
