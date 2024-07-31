from src.args.base_args import BaseExperimentArgsModel
from typing import Literal


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
