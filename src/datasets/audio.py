from typing import Literal
from torch.utils.data import Dataset
from datasets import DatasetDict
from src.args.wav2vec_args import AudioWav2VecArgsModel
from src.datasets.base_dataset import Sample, SampleBatch
from src.args.yaml_config import YamlConfigModel
import torch
from torch.nn.functional import pad
import re
from transformers import AutoTokenizer, PreTrainedTokenizer


class AudioDataset(Dataset):
    def __init__(
        self,
        hugg_dataset: DatasetDict,
        config: AudioWav2VecArgsModel,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        super().__init__()
        self.config = config
        self._data = hugg_dataset["test" if split == "val" else split]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        row = self._data[idx]
        return (
            torch.tensor(row["audio"]["array"], dtype=torch.float32),
            row["transcription"].upper(),
        )

    def get_collate_fn(self, tokenizer: PreTrainedTokenizer):
        def _collate(batch: list[Sample]):
            max_audio_len = max([x.size(0) for x, _ in batch])
            padded_audio = [
                pad(
                    x,
                    (0, max_audio_len - x.size(0)),
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

            batch_label_ids: torch.Tensor = tokenizer(
                [process_label(label) for _, label in batch],
                padding="longest",
                return_tensors="pt",
            ).input_ids

            return SampleBatch(torch.stack(padded_audio), batch_label_ids)

        return _collate
