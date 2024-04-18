from typing import Literal, Type, Optional
from numpy import pad
import torch
import torch.nn.functional as F
from torch import log_softmax, nn, Tensor
import math
from torch.nn.modules import (
    MultiheadAttention,
    Linear,
    Dropout,
    BatchNorm1d,
    TransformerEncoderLayer,
)
from src.args.base_args import B2TArgsModel
from src.datasets.brain2text import B2tSampleBatch
from src.model.b2tmodel import B2TModel, ModelOutput
from pydantic import BaseModel


class B2TMvtsTransformerArgsModel(BaseModel):
    dim_feedforward: int = 256
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.5
    dim_model: int = 1024
    norm: Literal["BatchNorm"] = "BatchNorm"
    classifier_activation: Literal["gelu"] = "gelu"
    freeze: bool = False
    pos_encoding: Literal["learnable"] = "learnable"


class MvtsTransformerModel(B2TModel):
    def __init__(
        self,
        config: B2TMvtsTransformerArgsModel,
        vocab_size: int,
        in_size: int,
        pad_token_id=0,
    ):
        super().__init__()
        self.config = config
        self.max_len = 1000
        self.pad_token_id = pad_token_id
        self.loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.transformer = TSTransformerEncoderClassiregressor(
            feat_dim=in_size,
            d_model=self.config.dim_model,
            max_len=self.max_len,
            n_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            num_classes=vocab_size,
            dropout=self.config.dropout,
            pos_encoding=self.config.pos_encoding,
            activation=self.config.classifier_activation,
            norm=self.config.norm,
            freeze=self.config.freeze,
        )

    def forward(self, batch: B2tSampleBatch):
        x, targets = batch

        device = x.device
        if targets is not None:
            targets = torch.where(
                targets == self.pad_token_id, torch.tensor(-100), targets
            )

        x_padded = torch.zeros(x.shape[0], self.max_len, x.shape[2])
        x_padded[:, : x.shape[1], :] = x
        mask = x_padded != 0
        sequence_mask = mask.any(-1)
        out = self.transformer.forward(x_padded.to(device), sequence_mask.to(device))

        if targets != None:
            target_lens = batch.target_lens
            input_lens = batch.input_lens
            if target_lens != None:
                ctc_loss = self.loss(
                    out.log_softmax(-1).transpose(0, 1),
                    targets,
                    input_lens.to(device),
                    target_lens.to(device),
                )

                if ctc_loss.item() < 0:
                    print(
                        f"\nWarning: loss is negative, this might be due to prediction lens ({input_lens.tolist()}) being smaller than target lens {target_lens.tolist()}\n"
                    )

                return ModelOutput(
                    out,
                    {"ctc_loss": ctc_loss.item()},
                    ctc_loss,
                )
        return ModelOutput(out, {}, None)


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    """Transformer encoder with BatchNorm instead of LayerNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multihead attention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(
            d_model, eps=1e-5
        )  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer.
            src_mask: the mask for the src sequence.
            src_key_padding_mask: the mask for the src keys per batch.
            is_causal: flag for causaility. Present for compatibility with nn.
        """
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class FixedPositionalEncoding(nn.Module):
    """Positional encoding of input.
    Args:
        d_model: the embed dim.
        dropout: the dropout value.
        max_len: the max. length of the incoming sequence.
        scale_factor: the scale factor for the positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1024,
        scale_factor: float = 1.0,
    ):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x: Tensor) -> Tensor:
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Positional encoding of input. This is learnable.

    Args:
        d_model: the embed dim.
        dropout: the dropout value.
        max_len: the max. length of the incoming sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(max_len, 1, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model.
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding: str) -> Type[nn.Module]:
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError(
        "pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding)
    )


class TSTransformerEncoder(nn.Module):
    """Time series transformer encoder module.

    Args:
        feat_dim: feature dimension
        max_len: maximum length of the input sequence
        d_model: the embed dim
        n_heads: the number of heads in the multihead attention models
        num_layers: the number of sub-encoder-layers in the encoder
        dim_feedforward: the dimension of the feedforward network model
        dropout: the dropout value
        pos_encoding: positional encoding method
        activation: the activation function of intermediate layer, relu or gelu
        norm: the normalization layer
        freeze: whether to freeze the positional encoding layer
    """

    def __init__(
        self,
        feat_dim: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        pos_encoding: str = "fixed",
        activation: str = "gelu",
        norm: str = "BatchNorm",
        freeze: bool = False,
    ):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(
            d_model, dropout=dropout * (1.0 - freeze), max_len=max_len
        )

        if norm == "LayerNorm":
            encoder_layer = TransformerEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = F.relu if activation == "relu" else F.gelu

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X: Tensor, padding_masks: Tensor) -> Tensor:
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model
        )  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(
            inp, src_key_padding_mask=~padding_masks
        )  # (seq_length, batch_size, d_model)
        output = self.act(
            output
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.

    Args:
        feat_dim: feature dimension
        max_len: maximum length of the input sequence
        d_model: the embed dim
        n_heads: the number of heads in the multihead attention models
        num_layers: the number of sub-encoder-layers in the encoder
        dim_feedforward: the dimension of the feedforward network model
        num_classes: the number of classes in the classification task
        dropout: the dropout value
        pos_encoding: positional encoding method
        activation: the activation function of intermediate layer, relu or gelu
        norm: the normalization layer
        freeze: whether to freeze the positional encoding layer
    """

    def __init__(
        self,
        feat_dim: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        dropout: float = 0.1,
        pos_encoding: str = "fixed",
        activation: str = "gelu",
        norm: str = "BatchNorm",
        freeze: bool = False,
    ):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(
            d_model, dropout=dropout * (1.0 - freeze), max_len=max_len
        )

        if norm == "LayerNorm":
            encoder_layer = TransformerEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model,
                self.n_heads,
                dim_feedforward,
                dropout * (1.0 - freeze),
                activation=activation,
            )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = F.relu if activation == "relu" else F.gelu

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.output_layer = nn.Linear(d_model, num_classes)

    # def build_output_module(self, d_model, max_len, num_classes):
    #     output_layer = nn.Linear(d_model * max_len, num_classes)
    #     # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
    #     # add F.log_softmax and use NLLoss
    #     return output_layer

    def forward(self, X: Tensor, padding_masks: Tensor) -> Tensor:
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model
        )  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        inverse_padding_masks = ~padding_masks
        output = self.transformer_encoder(
            inp, src_key_padding_mask=inverse_padding_masks.float()
        )  # (seq_length, batch_size, d_model)
        output = self.act(
            output
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = self.output_layer(output)  # (batch_size, seq_length, vocab)

        return output
