from typing import Any
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
from pathlib import Path

from .tokenizer import get_tokenizer


class Brain2TextDataset(Dataset):
    def __init__(
        self,
        ds_folder_path: str = "/hpi/fs00/scratch/florian.mueller/data/competitionData/train",
    ) -> None:
        super().__init__()

        if not os.path.exists(ds_folder_path):
            raise Exception(f"{ds_folder_path} does not exist.")

        data_files = [
            loadmat(Path(ds_folder_path) / fileName)
            for fileName in os.listdir(ds_folder_path)
        ]

        self.tokenizer = get_tokenizer()

        self.encoded_sentences = []
        self.brain_data_samples: list[torch.Tensor] = []

        for data_file in data_files:
            sentences: list[str] = data_file["sentenceText"]
            brain_data = data_file["spikePow"][0]

            for data_sample, sentence in zip(brain_data, sentences):
                self.brain_data_samples.append(torch.from_numpy(data_sample))
                self.encoded_sentences.append(self.tokenizer.encode(sentence))

        assert len(self.encoded_sentences) == len(self.brain_data_samples)

    def __len__(self):
        return len(self.encoded_sentences)

    def __getitem__(self, index) -> Any:
        return self.brain_data_samples[index], self.encodedSentences[index]

    def getTokenizer(self):
        return self.tokenizer
