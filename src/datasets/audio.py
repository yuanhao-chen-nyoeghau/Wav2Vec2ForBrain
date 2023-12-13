from typing import Literal
from torch.utils.data import Dataset
from datasets import DatasetDict
from args.yaml_config import YamlConfigModel
import torch


class AudioDataset(Dataset):
    def __init__(
        self,
        hugg_dataset: DatasetDict,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        super().__init__()
        self._data = hugg_dataset[split]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        row = self._data[idx]
        return torch.tensor(row["audio"]), row["text"].upper()
