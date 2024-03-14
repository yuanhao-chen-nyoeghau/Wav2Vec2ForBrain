from pydantic import BaseModel
from src.datasets.brain2text import B2tSampleBatch
from src.args.wav2vec_args import ACTIVATION_FUNCTION
from src.model.b2tmodel import B2TModel, ModelOutput
from typing import Literal, cast
import torch
from torch import nn
from torch.nn.functional import log_softmax
from src.util.nn_helper import compute_ctc_loss, create_fully_connected
import math
from torch.nn import functional as F


class GruArgsModel(BaseModel):
    hidden_size: int = 256
    bidirectional: bool = True
    num_gru_layers: int = 2
    bias: bool = True
    dropout: float = 0.0
    learnable_inital_state: bool = False
    classifier_hidden_sizes: list[int] = []
    classifier_activation: ACTIVATION_FUNCTION = "gelu"
    input_layer_nonlinearity: Literal["softsign"] = "softsign"
    unfolder_kernel_len: int = 14
    unfolder_stride_len: int = 4
    gaussian_smooth_width: int = 0


# Source: https://github.com/cffan/neural_seq_decoder/blob/master/src/neural_decoder/augmentations.py#L27
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size: int, sigma: float, dim=2):
        super(GaussianSmoothing, self).__init__()

        kernel_sizes = [kernel_size] * dim
        sigmas = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_sizes]
        )
        for size, std, mgrid in zip(kernel_sizes, sigmas, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(cast(torch.Tensor, kernel))

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")


class GRUModel(B2TModel):
    def __init__(self, config: GruArgsModel, vocab_size: int, pad_token_id=0):
        super().__init__()
        self.config = config
        self.num_directions = 2 if config.bidirectional else 1
        self.pad_token_id = pad_token_id
        self.hidden_start = nn.Parameter(
            torch.randn(
                self.num_directions * config.num_gru_layers,
                config.hidden_size,
                requires_grad=True,
            )
        )
        if config.input_layer_nonlinearity != "softsign":
            raise NotImplementedError(
                "Only softsign is currently supported as input layer nonlinearity"
            )
        self.input_layer_nonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (config.unfolder_kernel_len, 1),
            dilation=1,
            padding=0,
            stride=config.unfolder_stride_len,
        )
        n_days = 24
        neural_dim_len = 256
        self.gaussian_smoother = GaussianSmoothing(
            neural_dim_len, 20, config.gaussian_smooth_width, dim=1
        )
        self.day_weights = torch.nn.Parameter(
            torch.randn(n_days, neural_dim_len, neural_dim_len)
        )
        self.day_bias = torch.nn.Parameter(torch.zeros(n_days, 1, neural_dim_len))
        for x in range(n_days):
            self.day_weights.data[x, :, :] = torch.eye(neural_dim_len)

        self.gru = torch.nn.GRU(
            (neural_dim_len) * config.unfolder_kernel_len,
            config.hidden_size,
            config.num_gru_layers,
            dropout=config.dropout,
            bias=config.bias,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        # Input layers
        for x in range(n_days):
            setattr(
                self, "inpLayer" + str(x), nn.Linear(neural_dim_len, neural_dim_len)
            )

        for x in range(n_days):
            layer = getattr(self, "inpLayer" + str(x))
            layer.weight = torch.nn.Parameter(layer.weight + torch.eye(neural_dim_len))

        self.classifier = create_fully_connected(
            config.hidden_size * self.num_directions,
            vocab_size,
            config.classifier_hidden_sizes,
            config.classifier_activation,
        )

        self.loss: nn.CTCLoss = nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=True
        )

    def forward(self, batch: B2tSampleBatch) -> ModelOutput:
        x, targets = batch
        day_idxs = batch.day_idxs

        assert targets is not None, "Targets must be set"
        if targets is not None:
            targets = torch.where(
                targets == self.pad_token_id, torch.tensor(-100), targets
            )

        batch_size = x.shape[0]

        neural_input = torch.permute(x, (0, 2, 1))
        neural_input = self.gaussian_smoother(neural_input)
        neural_input = torch.permute(neural_input, (0, 2, 1))

        # apply day layer
        day_weights = torch.index_select(self.day_weights, 0, day_idxs)
        transformed_neural = torch.einsum(
            "btd,bdk->btk", neural_input, day_weights
        ) + torch.index_select(self.day_bias, 0, day_idxs)
        transformed_neural = self.input_layer_nonlinearity(transformed_neural)

        # stride/kernel
        strided_inputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformed_neural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        out, h_out = (
            self.gru(
                strided_inputs, self.hidden_start.unsqueeze(1).repeat(1, batch_size, 1)
            )
            if self.config.learnable_inital_state
            else self.gru(strided_inputs)
        )

        # out shape: (batch_size, seq_len, hidden_size * num_directions)
        out = self.classifier(out)

        # out shape: (batch_size, seq_len, vocab_size)

        out = log_softmax(out, -1)
        # TODO: check if ctc loss calculation is still valid
        ctc_loss = self.loss.forward(
            torch.permute(out, [1, 0, 2]),
            targets,
            (
                (batch.input_lens - self.config.unfolder_kernel_len)
                / self.config.unfolder_stride_len
            ).to(torch.int32),
            batch.target_lens,
        )

        return ModelOutput(
            out,
            {"ctc_loss": ctc_loss.item()},
            ctc_loss,
        )
