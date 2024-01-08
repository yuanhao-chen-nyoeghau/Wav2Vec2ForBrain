from typing import Literal

from pydantic import Field
from src.args.base_args import BaseExperimentArgsModel


class B2TResnetArgsModel(BaseExperimentArgsModel):
    tokenizer: Literal["wav2vec_pretrained", "ours"] = "wav2vec_pretrained"
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h", "facebook/wav2vec2-base-960h"
    ] = "facebook/wav2vec2-base-960h"
    remove_punctuation: bool = True
    unfreeze_strategy: Literal["all", "classifier"] = "classifier"
