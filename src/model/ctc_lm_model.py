from src.datasets.base_dataset import SampleBatch
from src.experiments.ctc_lm_experiment import CtcLmArgsModel
from src.model.b2tmodel import B2TModel, ModelOutput
from typing import Optional
import torch
from torch import nn
from transformers import PreTrainedTokenizer
from torch.nn.functional import log_softmax

from src.util.nn_helper import compute_ctc_loss, create_fully_connected


class CTCLMModel(B2TModel):
    def __init__(self, config: CtcLmArgsModel, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.ctclm_d_model, nhead=config.ctclm_nhead, batch_first=True
        )
        self.tok_embed = nn.Linear(tokenizer.vocab_size, config.ctclm_d_model)
        self.pos_embed = nn.Embedding(2000, config.ctclm_d_model)
        self.norm = nn.LayerNorm(config.ctclm_d_model)

        self.transformer = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=config.ctclm_num_layers),
            create_fully_connected(
                config.ctclm_d_model,
                tokenizer.vocab_size,
                config.ctclm_classifier_hidden_sizes,
                config.ctclm_classifier_activation,
            ),
        )
        self.loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(self, batch: SampleBatch) -> ModelOutput:
        x, targets = batch
        assert targets is not None, "Targets must be set"
        device = x.device
        if targets is not None:
            targets = torch.where(
                targets == self.tokenizer.pad_token_id, torch.tensor(-100), targets
            )
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(device)
        batch_size = x.size(0)
        pos = pos.expand(batch_size, -1)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos)

        out = self.transformer(self.norm(embedding))
        ctc_loss_before = compute_ctc_loss(x, x.log(), targets, self.loss)
        out = log_softmax(out, -1)
        # out shape: (seq_len, batch_size, vocab_size)
        ctc_loss_after = compute_ctc_loss(x, out, targets, self.loss)

        return ModelOutput(
            out,
            {
                "ctc_loss": ctc_loss_after.item(),
                "ctc_loss_input_seq": ctc_loss_before.item(),
                "ctc_loss_diff": ctc_loss_before.item() - ctc_loss_after.item(),
            },
            ctc_loss_after,
        )
