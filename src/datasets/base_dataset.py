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

    def cuda(self):
        copy = self._replace(input=self.input.cuda(), target=self.target.cuda())
        # Putting all tensors of subclass attributes to cuda
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                copy.__setattr__(key, value.cuda())
            else:
                copy.__setattr__(key, value)
        return copy


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
