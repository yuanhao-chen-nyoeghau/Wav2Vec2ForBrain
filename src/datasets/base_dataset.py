from collections import namedtuple
from torch.utils.data import Dataset
from abc import abstractclassmethod, abstractmethod
import torch
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

from src.datasets.batch_types import SampleBatch


class Sample(NamedTuple):
    input: torch.Tensor
    target: Any


class BaseDataset(Dataset):
    @abstractmethod
    def __getitem__(self, index) -> Sample:
        raise NotImplementedError(
            "This method should be overridden in a subclass of BaseDataset"
        )

    @abstractmethod
    def get_collate_fn(self) -> Callable[[list[Sample]], SampleBatch]:
        raise NotImplementedError(
            "This method should be overridden in a subclass of BaseDataset"
        )
