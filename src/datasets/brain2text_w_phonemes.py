from typing import Callable, Literal, NamedTuple
from attr import dataclass
import torch
from src.datasets.batch_types import PhonemeSampleBatch
from src.datasets.brain2text import B2tSample, Brain2TextDataset
import re
from g2p_en import G2p
from torch.nn.functional import pad
import re
from src.util.phoneme_helper import get_phoneme_seq
from src.util.phoneme_helper import PHONE_DEF_SIL


class PhonemeSample(B2tSample):
    transcription: str
    phonemes: list[str]


class Brain2TextWPhonemesDataset(Brain2TextDataset):
    vocab_size = len(PHONE_DEF_SIL) + 1
    vocab = ["blank"] + PHONE_DEF_SIL
    from src.args.base_args import B2TDatasetArgsModel
    from src.args.yaml_config import YamlConfigModel

    def __init__(
        self,
        config: B2TDatasetArgsModel,
        yaml_config: YamlConfigModel,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        super().__init__(config, yaml_config, split)

        g2p = G2p()
        self.phoneme_seqs = [
            get_phoneme_seq(g2p, sample.target) for sample in self.samples
        ]

    def __getitem__(self, index: int) -> PhonemeSample:
        sample: B2tSample = super().__getitem__(index)
        phoneme_ids, phonemes = self.phoneme_seqs[index]

        if self.config.remove_punctuation:
            chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'
            transcription = re.sub(chars_to_ignore_regex, "", sample.target)

        day_idx = sample.day_idx

        sample = PhonemeSample(sample.input, phoneme_ids)
        sample.day_idx = day_idx
        sample.transcription = transcription
        sample.phonemes = phonemes
        return sample

    def get_collate_fn(self) -> Callable[[list[PhonemeSample]], PhonemeSampleBatch]:
        multiple_channels = (
            self.config.preprocessing == "seperate_zscoring_2channels"
            or self.config.preprocessing == "seperate_zscoring_4channels"
        )

        def _collate(samples: list[PhonemeSample]):
            max_block_len = max(
                [x.size(1 if multiple_channels else 0) for x, _ in samples]
            )
            padded_blocks = [
                pad(
                    x,
                    (0, 0, 0, max_block_len - x.size(1 if multiple_channels else 0)),
                    mode="constant",
                    value=0,
                )
                for x, _ in samples
            ]

            max_phone_seq_len = max([len(phoneme_ids) for _, phoneme_ids in samples])
            padded_phoneme_ids = [
                pad(
                    torch.tensor(phoneme_ids),
                    (0, max_phone_seq_len - len(phoneme_ids)),
                    mode="constant",
                    value=0,
                )
                for _, phoneme_ids in samples
            ]

            batch = PhonemeSampleBatch(
                torch.stack(padded_blocks),
                torch.stack(padded_phoneme_ids),
            )
            batch.day_idxs = torch.tensor([sample.day_idx for sample in samples])
            batch.transcriptions = [sample.transcription for sample in samples]
            batch.phonemes = [sample.phonemes for sample in samples]
            batch.target_lens = torch.tensor(
                [len(phoneme_ids) for _, phoneme_ids in samples]
            )
            batch.input_lens = torch.tensor([x.size(0) for x, _ in samples])
            return batch

        return _collate
