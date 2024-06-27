import re
from typing import Dict, Literal, cast
from src.datasets.brain2text import B2tSample
from src.datasets.base_dataset import Sample
from src.datasets.batch_types import PhonemeSampleBatch, SampleBatch
from torch.nn.functional import pad

from src.datasets.audio import AudioDataset
from src.datasets.brain2text_w_phonemes import (
    PHONE_DEF,
    PHONE_DEF_SIL,
    SIL_DEF,
    PhonemeSeq,
    PhonemeSample,
)
from datasets import DatasetDict
from transformers import PreTrainedTokenizer
from g2p_en import G2p
import torch

from args.wav2vec_args import AudioWav2VecArgsModel


class AudioWPhonemesDataset:
    def __init__(
        self,
        hugg_dataset: DatasetDict,
        config: AudioWav2VecArgsModel,
        tokenizer: PreTrainedTokenizer,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        self.config = config
        self._data = hugg_dataset["test" if split == "val" else split]
        self.tokenizer = tokenizer
        self.g2p = G2p()
        self.phoneme_seqs = [
            self.get_phoneme_seq(cast(Dict, sample)["transcription"])
            for sample in self._data
        ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> PhonemeSample:
        row = self._data[index]
        sample: B2tSample = B2tSample(
            torch.tensor(row["audio"]["array"], dtype=torch.float32),
            row["transcription"].upper(),
        )
        phoneme_ids, phonemes = self.phoneme_seqs[index]

        if self.config.remove_punctuation:
            chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'
            transcription = re.sub(chars_to_ignore_regex, "", sample.target)

        sample = PhonemeSample(sample.input, phoneme_ids)
        sample.transcription = transcription
        sample.phonemes = phonemes
        return sample

    def get_collate_fn(self):
        def _collate(samples: list[PhonemeSample]):
            max_audio_len = max([x.size(0) for x, _ in samples])
            padded_audio = [
                pad(
                    x,
                    (0, max_audio_len - x.size(0)),
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
                torch.stack(padded_audio),
                torch.stack(padded_phoneme_ids),
            )
            batch.transcriptions = [sample.transcription for sample in samples]
            batch.phonemes = [sample.phonemes for sample in samples]
            batch.target_lens = torch.tensor(
                [len(phoneme_ids) for _, phoneme_ids in samples]
            )
            batch.input_lens = torch.tensor([x.size(0) for x, _ in samples])

            return batch

        return _collate

    def get_phoneme_seq(self, transcription: str) -> PhonemeSeq:

        def phoneToId(p):
            return PHONE_DEF_SIL.index(p)

        phonemes = []
        if len(transcription) == 0:
            phonemes = SIL_DEF
        else:
            for p in self.g2p(transcription.replace("<s>", "").replace("</s>", "")):
                if p == " ":
                    phonemes.append("SIL")
                p = re.sub(r"[0-9]", "", p)  # Remove stress
                if re.match(r"[A-Z]+", p):  # Only keep phonemes
                    phonemes.append(p)
            # add one SIL symbol at the end so there's one at the end of each word
            phonemes.append("SIL")

        phoneme_ids = [
            phoneToId(p) + 1 for p in phonemes
        ]  # +1 to shift the ids by 1 as 0 is blank
        return PhonemeSeq(phoneme_ids, phonemes)
