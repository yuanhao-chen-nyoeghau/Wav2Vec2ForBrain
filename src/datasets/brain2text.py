from typing import Any
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from transformers import Wav2Vec2CTCTokenizer, AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from .tokenizer import getTokenizer


class Brain2TextDataset(Dataset):
    def __init__(
        self,
        dsFolderPath: str = "/hpi/fs00/scratch/florian.mueller/data/competitionData/train",
    ) -> None:
        super().__init__()

        if not os.path.exists(dsFolderPath):
            raise Exception(f"{dsFolderPath} does not exist.")

        dataFiles = [
            loadmat(Path(dsFolderPath) / fileName)
            for fileName in os.listdir(dsFolderPath)
        ]

        self.tokenizer = getTokenizer()

        self.encodedSentences = []
        self.brainDataSamples: list[torch.Tensor] = []

        for dataFile in dataFiles:
            blockSentences: list[str] = dataFile["sentenceText"]
            blockBrainData = dataFile["spikePow"][0]

            for dataSample, sentence in zip(blockBrainData, blockSentences):
                self.brainDataSamples.append(torch.from_numpy(dataSample))
                self.encodedSentences.append(self.tokenizer.encode(sentence))

        assert len(self.encodedSentences) == len(self.brainDataSamples)

    def __len__(self):
        return len(self.encodedSentences)

    def __getitem__(self, index) -> Any:
        return self.brainDataSamples[index], self.encodedSentences[index]

    def getTokenizer(self):
        return self.tokenizer
