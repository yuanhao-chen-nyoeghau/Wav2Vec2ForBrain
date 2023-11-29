from typing import Any, Literal
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
from pathlib import Path
import torch
import numpy as np
from src.args.yaml_config import YamlConfigModel
from src.args.base_args import BaseArgsModel
from src.datasets.tokenizer import get_tokenizer


class Brain2TextDataset(Dataset):
    def __init__(
        self,
        config: BaseArgsModel,
        yaml_config: YamlConfigModel,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        super().__init__()
        if split == "test" and config.competition_mode:
            split = "val"

        if split == "val":
            data_path = Path(yaml_config.dataset_splits_dir) / "test"
        else:
            data_path = Path(yaml_config.dataset_splits_dir) / "train"

        if not os.path.exists(data_path):
            raise Exception(f"{data_path} does not exist.")

        data_files = [
            loadmat(data_path / fileName) for fileName in os.listdir(data_path)
        ]

        self.tokenizer = get_tokenizer(
            dataset_splits_dir=yaml_config.dataset_splits_dir,
            cache_dir=yaml_config.cache_dir,
            max_token_length=1,
            vocab_size=256,
        )

        self.encoded_sentences: list[str] = []
        self.brain_data_samples: list[torch.Tensor] = []

        for dataFile in data_files:
            n_trials = dataFile["sentenceText"].shape[0]
            input_features = []
            transcriptions = []
            frame_lens = []

            # collect area 6v tx1 and spikePow features
            for i in range(n_trials):
                # get time series of TX and spike power for this trial
                # first 128 columns = area 6v only
                features = np.concatenate(
                    [
                        dataFile["tx1"][0, i][:, 0:128],
                        dataFile["spikePow"][0, i][:, 0:128],
                    ],
                    axis=1,
                )

                sentence_len = features.shape[0]
                sentence = dataFile["sentenceText"][i].strip()

                input_features.append(features)
                transcriptions.append(sentence)
                frame_lens.append(sentence_len)

            # block-wise feature normalization
            blockNums = np.squeeze(dataFile["blockIdx"])
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

            if config.preprocessing == "per_feature_normal_dist":
                for b in range(len(blocks)):
                    feats = np.concatenate(
                        input_features[blocks[b][0] : (blocks[b][-1] + 1)], axis=0
                    )
                    feats_mean = np.mean(feats, axis=0, keepdims=True)
                    feats_std = np.std(feats, axis=0, keepdims=True)
                    for i in blocks[b]:
                        input_features[i] = (input_features[i] - feats_mean) / (
                            feats_std + 1e-8
                        )

            for dataSample in input_features:
                self.brain_data_samples.append(torch.from_numpy(dataSample))
            for sentence in transcriptions:
                self.encoded_sentences.append(self.tokenizer.encode(sentence))

        assert len(self.encoded_sentences) == len(
            self.brain_data_samples
        ), "Length of labels and data samples must be equal."

    def __len__(self):
        return len(self.encoded_sentences)

    def __getitem__(self, index) -> Any:
        return self.brain_data_samples[index], self.encoded_sentences[index]

    def getTokenizer(self):
        return self.tokenizer
