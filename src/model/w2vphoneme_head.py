from typing import Optional, cast
from pydantic import BaseModel
import torch
from src.datasets.batch_types import B2tSampleBatch
from src.model.b2tmodel import B2TModel, ModelOutput
from src.util.nn_helper import ACTIVATION_FUNCTION, create_fully_connected
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    CausalLMOutput,
)


class W2VPhonemeHeadArgs(BaseModel):
    prepend_gru_to_head: bool = False
    head_gru_hidden_size: Optional[int] = None
    head_gru_num_layers: Optional[int] = None
    head_gru_dropout: Optional[float] = None
    head_fc_hidden_sizes: list[int] = []
    head_fc_activation_function: ACTIVATION_FUNCTION = "gelu"


class W2VPhonemeHead(torch.nn.Module):
    def __init__(self, config: W2VPhonemeHeadArgs, out_size: int):
        super().__init__()
        n_ipa_phonemes = 392

        if config.prepend_gru_to_head != True:
            assert (
                config.head_gru_num_layers is None
                and config.head_gru_hidden_size is None
                and config.head_gru_dropout is None
            ), "If prepend_gru_to_head is not set to True, head_gru_num_layers, head_gru_hidden_size and head_gru_dropout must be None"
            self.fc = create_fully_connected(
                n_ipa_phonemes,
                out_size,
                activation=config.head_fc_activation_function,
                hidden_sizes=config.head_fc_hidden_sizes,
            )
            self.gru = None
        else:
            assert (
                config.head_gru_num_layers is not None
                and config.head_gru_hidden_size is not None
            ), "If prepend_gru_to_head is set to True, head_gru_num_layers, head_gru_hidden_size and head_gru_dropout must be set"

            self.gru = torch.nn.GRU(
                n_ipa_phonemes,
                config.head_gru_hidden_size,
                config.head_gru_num_layers,
                dropout=(
                    config.head_gru_dropout
                    if config.head_gru_dropout is not None
                    else 0.0
                ),
                bias=True,
                bidirectional=True,
                batch_first=True,
            )
            self.fc = create_fully_connected(
                config.head_gru_hidden_size * 2,
                out_size,
                activation=config.head_fc_activation_function,
                hidden_sizes=config.head_fc_hidden_sizes,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor
        if self.gru is not None:
            out, _ = self.gru.forward(x)
        else:
            out = x
        out = self.fc(out)
        return out


class W2VPhonemeWithCustomHead(B2TModel):
    def __init__(
        self,
        wav2vec_checkpoint: str,
        head: W2VPhonemeHead,
    ):
        super().__init__()
        w2v_config = cast(
            Wav2Vec2Config,
            Wav2Vec2Config.from_pretrained(wav2vec_checkpoint),
        )
        self.w2v = cast(
            Wav2Vec2ForCTC,
            Wav2Vec2ForCTC.from_pretrained(
                wav2vec_checkpoint,
                config=w2v_config,
            ),
        )
        self.head = head
        self.loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(self, batch: B2tSampleBatch):
        x, target = batch
        assert (
            target is not None
            and batch.input_lens is not None
            and batch.target_lens is not None
        )
        out = cast(CausalLMOutput, self.w2v.forward(x, return_dict=True))
        out = self.head.forward(out.logits)

        feature_extract_output_lens = cast(
            torch.LongTensor,
            self.w2v._get_feat_extract_output_lengths(
                cast(torch.LongTensor, batch.input_lens.long())
            ),
        )
        ctc_loss = self.loss.forward(
            torch.log_softmax(out, -1).transpose(0, 1),
            target,
            feature_extract_output_lens.cuda(),
            batch.target_lens.cuda(),
        )
        metrics = {}
        if ctc_loss is not None:
            metrics["ctc_loss"] = ctc_loss.item()
        return ModelOutput(logits=out, metrics=metrics, loss=ctc_loss)
