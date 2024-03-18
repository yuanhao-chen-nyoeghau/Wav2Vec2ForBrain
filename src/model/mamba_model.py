from pydantic import BaseModel
from src.datasets.batch_types import B2tSampleBatch
from src.args.wav2vec_args import ACTIVATION_FUNCTION
from src.experiments.b2t_experiment import B2TExperiment
from src.args.base_args import (
    B2TArgsModel,
)
from src.model.b2tmodel import B2TModel, ModelOutput
from src.args.yaml_config import YamlConfigModel
from typing import Optional
import torch
from torch import nn
from transformers import PreTrainedTokenizer
from torch.nn.functional import log_softmax
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.models.mixer_seq_simple import MixerModel, _init_weights
from functools import partial
from collections import namedtuple
from src.util.nn_helper import compute_ctc_loss, create_fully_connected


class MambaArgsModel(BaseModel):
    mamba_d_model: int = 1024  # 2560
    mamba_n_layer: int = 32
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    classifier_hidden_sizes: list[int] = []  # [256, 128, 64]
    classifier_activation: ACTIVATION_FUNCTION = "gelu"
    feature_extractor_hidden_sizes: list[int] = []  # [256, 256, 128, 64]
    feature_extractor_activation: ACTIVATION_FUNCTION = "sigmoid"
    input_dropout: float = 0.0


class MambaLMHeadModel(B2TModel, GenerationMixin):
    def __init__(self, config: MambaArgsModel, vocab_size: int, in_size: int) -> None:
        self.config = config
        d_model = config.mamba_d_model
        n_layer = config.mamba_n_layer

        out_size = vocab_size
        ssm_cfg = None
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        factory_kwargs = {"device": "cuda", "dtype": torch.float32}
        initializer_cfg = None

        super().__init__()
        # if in_size % pad_vocab_size_multiple != 0:
        #     in_size += pad_vocab_size_multiple - (in_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=d_model,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.backbone.embedding = nn.Sequential(create_fully_connected(in_size, d_model, config.feature_extractor_hidden_sizes, config.feature_extractor_activation), nn.Dropout(config.input_dropout))  # type: ignore

        self.lm_head = create_fully_connected(
            d_model,
            out_size,
            config.classifier_hidden_sizes,
            config.classifier_activation,
        )  # nn.Linear(d_model, out_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        # self.tie_weights()

    def tie_weights(self):
        pass
        # self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)


class MambaModel(B2TModel):
    def __init__(
        self, config: MambaArgsModel, vocab_size: int, in_size: int, pad_token_id=0
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.model = MambaLMHeadModel(config, vocab_size, in_size)
        self.loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(self, batch: B2tSampleBatch) -> ModelOutput:
        x, targets = batch
        assert targets is not None, "Targets must be set"
        device = targets.device
        if targets is not None:
            targets = torch.where(
                targets == self.pad_token_id, torch.tensor(-100), targets
            )

        # x shape: (batch_size, seq_len, 256)
        out = self.model.forward(x).logits

        # out shape: (batch_size, seq_len, vocab_size)
        out = log_softmax(out, -1)
        ctc_loss = compute_ctc_loss(x, out, targets, self.loss)

        return ModelOutput(
            out,
            {"ctc_loss": ctc_loss.item()},
            ctc_loss,
        )
