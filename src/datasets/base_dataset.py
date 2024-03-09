from collections import namedtuple
from torch.utils.data import Dataset
from abc import abstractclassmethod, abstractmethod
import torch
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple


class SampleBatch(NamedTuple):
    input: torch.Tensor
    target: (
        torch.Tensor
    )  # Batch of tokenized targets (i.e. a batch of lists of target ids)


class Sample(NamedTuple):
    input: torch.Tensor
    target: Any


class BaseDataset(Dataset):
    @abstractmethod
    def __getitem__(self, index) -> Sample:
        raise NotImplementedError(
            "This method should be overridden in a subclass of BaseDataset"
        )

    @classmethod
    def get_collate_fn(cls) -> Callable[[list[Sample]], SampleBatch]:
        raise NotImplementedError(
            "This method should be overridden in a subclass of BaseDataset"
        )
