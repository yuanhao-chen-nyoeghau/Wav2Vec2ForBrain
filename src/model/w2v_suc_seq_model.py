from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Config,
)
import torch
from torch import log_softmax, nn
from typing import Literal, cast

from src.datasets.timit_seq_dataset import TimitSeqSampleBatch
from src.model.w2v_no_encoder import Wav2Vec2WithoutTransformerModel
from src.model.suc import SUCModel
from src.model.w2v_suc_model import W2VSUCArgsModel
from src.args.wav2vec_args import ACTIVATION_FUNCTION
from src.datasets.batch_types import PhonemeSampleBatch
from src.datasets.brain2text_w_phonemes import PHONE_DEF_SIL
from src.model.b2tmodel import B2TModel, ModelOutput


class W2VSUC_CTCArgsModel(W2VSUCArgsModel):
    loss_function: Literal["ctc"] = "ctc"
    gru_hidden_size: int = 256
    bidirectional: bool = True
    num_gru_layers: int = 2
    bias: bool = True
    dropout: float = 0.0
    learnable_inital_state: bool = False
    fc_hidden_sizes: list[int] = []
    fc_activation_function: ACTIVATION_FUNCTION = "gelu"


class SUCCTCHead(nn.Module):
    def __init__(self, config: W2VSUC_CTCArgsModel):
        super().__init__()
        self.gru = torch.nn.GRU(
            len(PHONE_DEF_SIL),
            config.gru_hidden_size,
            config.num_gru_layers,
            dropout=config.dropout,
            bias=config.bias,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        self.gru_project = nn.Linear(
            config.gru_hidden_size * (2 if config.bidirectional else 1),
            len(PHONE_DEF_SIL) + 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_output, _ = self.gru(x)
        out = self.gru_project(gru_output)
        return out


class SUCForCTC(nn.Module):
    def __init__(self, config: W2VSUC_CTCArgsModel):
        super().__init__()
        self.suc = SUCModel(config.suc_hidden_sizes, config.suc_hidden_activation)
        self.ctc_head = SUCCTCHead(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        suc_output = self.suc(x)
        return self.ctc_head(suc_output)


class W2VSUCSeqModel(B2TModel):
    def __init__(self, config: W2VSUC_CTCArgsModel):
        super().__init__()
        self.config = config
        w2v_config = cast(
            Wav2Vec2Config,
            Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h"),
        )
        self.w2v_feature_extractor = cast(
            Wav2Vec2WithoutTransformerModel,
            Wav2Vec2WithoutTransformerModel.from_pretrained(
                "facebook/wav2vec2-base-960h", config=w2v_config
            ),
        )
        self.dropout = nn.Dropout(config.suc_dropout)
        self.suc_for_ctc = SUCForCTC(config)

        self.loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(self, batch: TimitSeqSampleBatch) -> ModelOutput:
        if batch.target is None:
            raise ValueError("Target is required for training")

        device = batch.input.device
        w2v_output = self.w2v_feature_extractor(batch.input)
        w2v_output = self.dropout(w2v_output)
        out = self.suc_for_ctc(w2v_output)

        feature_extract_output_lens = cast(
            torch.LongTensor,
            self.w2v_feature_extractor._get_feat_extract_output_lengths(
                cast(torch.LongTensor, batch.input_lens.long())
            ),
        )

        assert type(self.loss) == nn.CTCLoss
        loss = self.loss.forward(
            log_softmax(out, -1).transpose(0, 1),
            batch.target,
            feature_extract_output_lens.to(device),
            batch.target_lens.to(device),
        )
        metrics = {"ctc_loss": loss.item()}

        return ModelOutput(
            out,
            metrics,
            loss=loss,
            logit_lens=feature_extract_output_lens,
        )
