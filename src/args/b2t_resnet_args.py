from typing import Literal

from pydantic import Field
from src.args.wav2vec_args import B2TWav2VecArgsModel
from src.args.base_args import B2TDatasetArgsModel, BaseExperimentArgsModel


class B2TWav2VecResnetArgsModel(B2TWav2VecArgsModel):
    mode: Literal["pretraining", "finetuning"] = "finetuning"
    conv_bias: bool = True
    conv_stride: list[int] = [1]  # [5, 2, 2, 2, 2, 2, 2]
    conv_kernel: list[int] = [1]  # [10, 3, 3, 3, 3, 2, 2]
    conv_dim: list[int] = [512]  # [512, 512, 512, 512, 512, 512, 512]
    feat_extract_activation: Literal[
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
    ] = "gelu"
    feat_extract_norm: Literal["group", "layer"] = "group"
    num_feat_extract_layers: int = 1  # 7
