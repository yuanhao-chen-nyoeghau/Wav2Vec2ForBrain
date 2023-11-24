from typing import Any, Literal
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
from pathlib import Path
import torch

from src.args.yaml_config import YamlConfigModel

from .tokenizer import get_tokenizer


class Brain2TextDataset(Dataset):
    def __init__(
        self,
        config: YamlConfigModel,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        super().__init__()

        if not os.path.exists(Path(config.dataset_splits_dir) / str(split)):
            raise Exception(
                f"{Path(config.dataset_splits_dir) / str(split)} does not exist."
            )

        data_files = [
            loadmat(Path(config.dataset_splits_dir) / split / fileName)
            for fileName in os.listdir(Path(config.dataset_splits_dir) / str(split))
        ]

        self.tokenizer = get_tokenizer(
            train_file=config.dataset_all_sentences_path,
            dataset_splits_dir=config.dataset_splits_dir,
            tokenizer_config_dir=config.tokenizer_config_dir,
            max_token_length=1,
            vocab_size=256,
        )

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
