from torch.nn import Module
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class ModelOutput:
    logits: torch.Tensor
    metrics: dict[str, float]
    loss: Optional[torch.Tensor] = None


class B2TModel(Module, ABC):
    @abstractmethod
    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        pass
