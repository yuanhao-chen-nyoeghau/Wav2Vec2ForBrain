from typing import Any
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
from pathlib import Path


class Brain2TextDataset(Dataset):
    def __init__(
        self,
        dsFolderPath: str = "/hpi/fs00/scratch/florian.mueller/data/competitionData/train",
        sentencesFolderPath: str = "/hpi/fs00/scratch/florian.mueller/data/sentences",
        limit: int | None = None,
    ) -> None:
        super().__init__()

        if not os.path.exists(dsFolderPath):
            raise Exception(f"{dsFolderPath} does not exist.")

        self.samples = [
            loadmat(Path(dsFolderPath) / fileName)
            for fileName in os.listdir(dsFolderPath)
        ]
        self.sentences = [
            loadmat(Path(sentencesFolderPath) / fileName)
            for fileName in os.listdir(sentencesFolderPath)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> Any:
        return self.samples[index], self.sentences[index]
