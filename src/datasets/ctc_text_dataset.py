from typing import Literal, Optional, cast
from torch.utils.data import Dataset
import os
import torch
from src.datasets.base_dataset import BaseDataset
from src.args.yaml_config import YamlConfigModel
from src.args.base_args import CTCTextDatasetArgsModel
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from random import Random, random
from math import floor
from torch.nn.functional import pad
import re


class CTCTextDataset(Dataset):
    def __init__(
        self,
        config: CTCTextDatasetArgsModel,
        yaml_config: YamlConfigModel,
        tokenizer: PreTrainedTokenizer,
        sentences: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.yaml_config = yaml_config
        self.tokenizer = tokenizer

        self.cache: dict[int, tuple[torch.Tensor, str]] = {}

        base_dir = os.path.join(yaml_config.cache_dir, "generics_kb_best")
        cache_dir = os.path.join(base_dir, "cache")
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        self.sentences: list[str]

        if sentences is None:
            self.sentences = cast(
                list[str],
                load_dataset(
                    "generics_kb",
                    "generics_kb_best",
                    cache_dir=cache_dir,
                    data_dir=data_dir,
                    split="train",
                )[  # type: ignore
                    "generic_sentence"
                ],
            )
            Random(42).shuffle(self.sentences)
        else:
            self.sentences = sentences

        ratio_sum = config.train_ratio + config.val_ratio + config.test_ratio
        assert (
            ratio_sum == 1.0
        ), f"The sum of train_ratio, val_ratio and test_ratio should be 1.0 but is {ratio_sum}"

    def __len__(self):
        return (
            len(self.sentences)
            if self.config.limit_samples is None
            else min(len(self.sentences), self.config.limit_samples)
        )

    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        if self.cache.get(index) is not None:
            return self.cache[index]
        sentence = self.sentences[index].upper()
        sentence = f"<s>{sentence}</s>"
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt").squeeze(0)  # type: ignore
        prob_seq = self.convert_ids_to_prob_seq(input_ids)
        if self.config.cache_generated_samples:
            self.cache[index] = prob_seq, sentence

        return prob_seq, sentence

    def convert_ids_to_prob_seq(
        self,
        id_seq: list[int],
    ):
        vocab_size = self.tokenizer.vocab_size
        c = self.config
        improved_seq = []
        # insert wrong characters
        for i in range(len(id_seq)):
            improved_seq.append(id_seq[i])
            if random() < c.insert_wrong_char_prob:
                improved_seq.append(floor(random() * (vocab_size)))

        # remove characters
        for i in range(len(improved_seq)):
            j = len(improved_seq) - i - 1
            if random() < c.remove_char_prob:
                improved_seq.pop(j)

        # insert blanks

        len_before = len(improved_seq)
        for i in range(len(improved_seq)):
            j = len_before - i
            while random() > (1 / (c.avg_num_blank_after_char + 1)):
                improved_seq.insert(j, 0)

        base = (
            torch.randn(len(improved_seq), vocab_size)
            .mul(c.noise_std)
            .add(c.noise_mean)
        )

        for i, id in enumerate(improved_seq):
            base[i][id] = -0.001

        for i, id in enumerate(improved_seq):
            if id > 4:
                if random() < c.correct_as_second_prob:
                    base[i][floor(random() * (vocab_size))] = -0.00001
                    base[i][id] = -1.0
            else:
                if random() < c.random_second_id_in_blank_prob:
                    base[i][floor(random() * (vocab_size))] = -1.0

        return base.softmax(-1)

    def get_split(self, split: Literal["train", "val", "test"]) -> Dataset:
        train_stop = int(len(self.sentences) * self.config.train_ratio)
        val_stop = train_stop + int(len(self.sentences) * self.config.val_ratio)

        sentences = self.sentences = (
            self.sentences[:train_stop]
            if split == "train"
            else (
                self.sentences[train_stop:val_stop]
                if split == "val"
                else self.sentences[val_stop:]
            )
        )
        return CTCTextDataset(self.config, self.yaml_config, self.tokenizer, sentences)

    def get_collate_fn(self):
        def _collate(batch: list[tuple[torch.Tensor, str]]):
            max_block_len = max([x.size(0) for x, _ in batch])
            padded_blocks = [
                pad(
                    x,
                    (0, 0, 0, max_block_len - x.size(0)),
                    mode="constant",
                    value=0,
                )
                for x, _ in batch
            ]

            def process_label(label: str) -> str:
                if self.config.remove_punctuation:
                    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'
                    label = re.sub(chars_to_ignore_regex, "", label)
                # label = label.upper()
                return label

            batch_label_ids: list[list[int]] = self.tokenizer(
                [process_label(label) for _, label in batch],
                padding="longest",
                return_tensors="pt",
            ).input_ids

            return torch.stack(padded_blocks), batch_label_ids

        return _collate
