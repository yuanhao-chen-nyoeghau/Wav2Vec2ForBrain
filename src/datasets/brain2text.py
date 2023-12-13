from typing import Any, Literal, Callable
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
from pathlib import Path
import torch
import numpy as np
from src.args.yaml_config import YamlConfigModel
from src.args.base_args import B2TDatasetArgsModel
from src.datasets.tokenizer import get_tokenizer
from src.datasets.preprocessing import (
    preprocess_competition_recommended,
    preprocess_seperate_zscoring,
    preprocess_only_spikepow_unnormalized,
    preprocess_only_spikepow_zscored,
    preprocess_only_tx_unnormalized,
    preprocess_only_tx_zscored,
)
from transformers import AutoTokenizer, PreTrainedTokenizer

PreprocessingFunctions: dict[
    str,
    Callable[[dict, list[np.ndarray[Any, np.dtype[np.int32]]]], tuple[list, list[str]]],
] = {
    "competition_recommended": preprocess_competition_recommended,
    "seperate_zscoring": preprocess_seperate_zscoring,
    "only_tx_unnormalized": preprocess_only_tx_unnormalized,
    "only_tx_zscored": preprocess_only_tx_zscored,
    "only_spikepow_unnormalized": preprocess_only_spikepow_unnormalized,
    "only_spikepow_zscored": preprocess_only_spikepow_zscored,
}


class Brain2TextDataset(Dataset):
    def __init__(
        self,
        config: B2TDatasetArgsModel,
        yaml_config: YamlConfigModel,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        super().__init__()
        self.config = config

        if split == "val":
            data_path = Path(yaml_config.dataset_splits_dir) / "test"
        elif split == "test" and config.competition_mode:
            data_path = Path(yaml_config.dataset_splits_dir) / "competitionHoldOut"
        else:
            data_path = Path(yaml_config.dataset_splits_dir) / "train"

        if not os.path.exists(data_path):
            raise Exception(f"{data_path} does not exist.")

        data_files = [
            loadmat(data_path / fileName) for fileName in os.listdir(data_path)
        ]

        self.transcriptions: list[str] = []
        self.brain_data_samples: list[torch.Tensor] = []
        preprocess = PreprocessingFunctions[config.preprocessing]

        for data_file in data_files:
            # block-wise feature normalization
            blockNums = np.squeeze(data_file["blockIdx"])
            blockList = np.unique(blockNums)

            if split == "test" and not config.competition_mode:
                blockList = [blockList[0]]
            if split == "train" and not config.competition_mode:
                blockList = blockList[1:]

            blocks = []
            for b in range(len(blockList)):
                sentIdx = np.argwhere(blockNums == blockList[b])
                sentIdx = sentIdx[:, 0].astype(np.int32)
                blocks.append(sentIdx)

            input_features, transcriptions = preprocess(data_file, blocks)

            for dataSample in input_features:
                self.brain_data_samples.append(
                    torch.tensor(dataSample, dtype=torch.float32)
                )
            for sentence in transcriptions:
                self.transcriptions.append(f"<s>{sentence.upper()}</s>")

        assert len(self.transcriptions) == len(
            self.brain_data_samples
        ), "Length of labels and data samples must be equal."

    def __len__(self):
        return (
            len(self.transcriptions)
            if self.config.limit_samples is None
            else self.config.limit_samples
        )

    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        return self.brain_data_samples[index], self.transcriptions[index]
