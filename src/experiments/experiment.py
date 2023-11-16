from abc import ABC, abstractmethod,abstractclassmethod
from torch.utils.data import Dataset
from argparse import ArgumentParser
from pydantic import BaseModel

class Experiment(ABC):
    def __init__(self, config: dict):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    # def get_dataset(self) -> Dataset:
    #     pass

    @abstractclassmethod
    def get_args_model() -> BaseModel:
        pass