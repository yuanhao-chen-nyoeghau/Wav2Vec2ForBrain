from typing import Literal
from attr import dataclass
import torch
from datasets.base_dataset import Sample
from src.datasets.brain2text import Brain2TextDataset
from src.args.yaml_config import YamlConfigModel
from src.args.base_args import B2TDatasetArgsModel
import re
from g2p_en import G2p

PHONE_DEF = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]

PHONE_DEF_SIL = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "SIL",
]

CHANG_PHONE_DEF = [
    "AA",
    "AE",
    "AH",
    "AW",
    "AY",
    "B",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "P",
    "R",
    "S",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
]

CONSONANT_DEF = [
    "CH",
    "SH",
    "JH",
    "R",
    "B",
    "M",
    "W",
    "V",
    "F",
    "P",
    "D",
    "N",
    "L",
    "S",
    "T",
    "Z",
    "TH",
    "G",
    "Y",
    "HH",
    "K",
    "NG",
    "ZH",
    "DH",
]
VOWEL_DEF = [
    "EY",
    "AE",
    "AY",
    "EH",
    "AA",
    "AW",
    "IY",
    "IH",
    "OY",
    "OW",
    "AO",
    "UH",
    "AH",
    "UW",
    "ER",
]

SIL_DEF = ["SIL"]


@dataclass
class PhonemeSeq:
    phoneme_ids: list[int]
    phonemes: list[str]

    def __iter__(self):
        return iter((self.phoneme_ids, self.phonemes))


class PhonemeSample(Sample):
    transcription: str
    phonemes: list[str]


class Brain2TextWPhonemesDataset(Brain2TextDataset):
    def __init__(
        self,
        config: B2TDatasetArgsModel,
        yaml_config: YamlConfigModel,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        super().__init__(config, yaml_config, split)

        self.g2p = G2p()
        self.phoneme_seqs = [
            self.get_phoneme_seq(transcription) for transcription in self.transcriptions
        ]

    def __getitem__(self, index) -> PhonemeSample:
        resampled, transcription = super().__getitem__(index)
        phonemes, phoneme_ids = self.phoneme_seqs[index]

        sample = PhonemeSample(resampled, phoneme_ids)
        sample.transcription = transcription
        sample.phonemes = phonemes
        return sample

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

        phoneme_ids = [phoneToId(p) + 1 for p in phonemes]
        return PhonemeSeq(phoneme_ids, phonemes)
