from torch.nn import Module
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
from src.datasets.batch_types import SampleBatch


@dataclass
class ModelOutput:
    logits: torch.Tensor
    metrics: dict[str, float]
    loss: Optional[torch.Tensor] = None


class B2TModel(Module, ABC):
    @abstractmethod
    def forward(self, batch: SampleBatch) -> ModelOutput:
        pass
