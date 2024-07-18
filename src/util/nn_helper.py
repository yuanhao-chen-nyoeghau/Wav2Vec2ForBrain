from sympy import false
from torch.nn import Linear, BatchNorm1d
from torch import batch_norm, nn
from transformers.activations import ACT2FN

from src.args.wav2vec_args import ACTIVATION_FUNCTION
import torch


def create_fully_connected(
    input_size: int,
    output_size: int,
    hidden_sizes=[],
    activation: ACTIVATION_FUNCTION = "gelu",
    use_batch_norm: bool = False,
):
    classifier_layers = []
    for i in range(-1, len(hidden_sizes)):
        is_last = i + 1 == len(hidden_sizes)
        is_first = i == -1
        in_size = input_size if is_first else hidden_sizes[i]
        out_size = output_size if is_last else hidden_sizes[i + 1]
        classifier_layers.append(Linear(in_size, out_size))
        if not is_last:
            if use_batch_norm:
                classifier_layers.append(BatchNorm1d(num_features=1))
            classifier_layers.append(ACT2FN[activation])
    return nn.Sequential(*classifier_layers)


def calc_seq_len(index_seq: torch.Tensor):
    for i in range(len(index_seq)):
        j = len(index_seq) - 1 - i
        if index_seq[j].item() > 0:
            return j + 1
    return 0


def compute_ctc_loss(
    model_input: torch.Tensor,
    out_log_softmaxed_batch: torch.Tensor,
    targets: torch.Tensor,
    loss: nn.CTCLoss,
    input_lens: torch.Tensor | None = None,
):
    """
    x: (batch_size, seq_len, *) - assuming all items of last dimension to be zero when padded
    out_log_softmaxed_batch: (batch_size, seq_len, vocab_size)
    """
    device = targets.device

    target_lens = torch.tensor([calc_seq_len(seq) for seq in targets])
    # out shape: (batch_size, seq_len, vocab_size)

    if input_lens is None:
        # non padded mask
        mask = model_input != 0
        # seq lens without padding
        # mask shape: (batch_size, seq_len, 256)
        input_lens = mask.any(-1)
        # input_lens shape: (batch_size, seq_len)
        input_lens = input_lens.sum(-1)
        # input_lens shape: (batch_size)
        input_lens = input_lens.clamp(max=out_log_softmaxed_batch.shape[1])
    out = out_log_softmaxed_batch.transpose(0, 1)
    # out shape: (seq_len, batch_size, vocab_size)
    ctc_loss = loss(
        out,
        targets,
        input_lens.to(device),
        target_lens.to(device),
    )
    if ctc_loss.item() < 0:
        print(
            f"\nWarning: loss is negative, this might be due to prediction lens ({input_lens.tolist()}) being smaller than target lens {target_lens.tolist()}\n"
        )
    return ctc_loss
