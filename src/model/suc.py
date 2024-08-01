import torch

from src.util.nn_helper import ACTIVATION_FUNCTION
from src.util.phoneme_helper import PHONE_DEF_SIL
from src.util.nn_helper import create_fully_connected


class SUCModel(torch.nn.Module):
    def __init__(
        self, hidden_sizes: list[int], activation: ACTIVATION_FUNCTION = "gelu"
    ):
        super().__init__()
        self.model = create_fully_connected(
            768, len(PHONE_DEF_SIL), hidden_sizes, activation=activation
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)
