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
    str, Callable[[dict, list[np.ndarray[np.int32]]], tuple[list, list[str]]]
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
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        super().__init__()

        split_dir = Path(yaml_config.dataset_splits_dir) / str(split)
        if not os.path.exists(split_dir):
            raise Exception(f"{split_dir} does not exist.")

        data_files = [
            loadmat(split_dir / fileName) for fileName in os.listdir(split_dir)
        ]

        self.tokenizer = tokenizer

        self.encoded_sentences: list[str] = []
        self.brain_data_samples: list[torch.Tensor] = []
        preprocess = PreprocessingFunctions[config.preprocessing]

        all_labels = torch.eye(self.tokenizer.__len__())
        for data_file in data_files:
            # block-wise feature normalization
            blockNums = np.squeeze(data_file["blockIdx"])
            blockList = np.unique(blockNums)
            blocks = []
            for b in range(len(blockList)):
                sentIdx = np.argwhere(blockNums == blockList[b])
                sentIdx = sentIdx[:, 0].astype(np.int32)
                blocks.append(sentIdx)
            input_features, transcriptions = preprocess(data_file, blocks)

            for dataSample in input_features:
                self.brain_data_samples.append(torch.from_numpy(dataSample))
            for sentence in transcriptions:
                self.encoded_sentences.append(
                    [
                        all_labels[token_id]
                        for token_id in self.tokenizer.encode(sentence)
                    ]
                )

        assert len(self.encoded_sentences) == len(
            self.brain_data_samples
        ), "Length of labels and data samples must be equal."

    def __len__(self):
        return len(self.encoded_sentences)

    def __getitem__(self, index) -> Any:
        return self.brain_data_samples[index], self.encoded_sentences[index]
