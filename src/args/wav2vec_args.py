from src.args.base_args import BaseExperimentArgsModel
from typing import Literal


class Wav2VecArgsModel(BaseExperimentArgsModel):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h", "facebook/wav2vec2-base-960h"
    ] = "facebook/wav2vec2-base-960h"
    brain2audioshape_strategy: Literal["shared_fc", "shared_fc_relu"] = "shared_fc"
    unfreeze_strategy: Literal[
        "wav2vec2featureextractor_ours", "all"
    ] = "wav2vec2featureextractor_ours"
