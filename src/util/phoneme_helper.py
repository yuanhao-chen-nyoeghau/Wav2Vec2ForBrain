from typing import NamedTuple
from g2p_en import G2p
import re

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

PHONE_DEF_SIL = PHONE_DEF + [
    "SIL",
]

SIL_DEF = ["SIL"]


class PhonemeSeq(NamedTuple):
    phoneme_ids: list[int]
    phonemes: list[str]


def get_phoneme_seq(g2p: G2p, transcription: str, zero_is_blank=True) -> PhonemeSeq:
    def phoneToId(p):
        return PHONE_DEF_SIL.index(p)

    phonemes = []
    if len(transcription) == 0:
        phonemes = SIL_DEF
    else:
        for p in g2p(transcription.replace("<s>", "").replace("</s>", "").upper()):
            if p == " ":
                phonemes.append("SIL")
            p = re.sub(r"[0-9]", "", p)  # Remove stress
            if re.match(r"[A-Z]+", p):  # Only keep phonemes
                phonemes.append(p)
        # add one SIL symbol at the end so there's one at the end of each word
        phonemes.append("SIL")

    phoneme_ids = (
        [phoneToId(p) + 1 for p in phonemes]
        if zero_is_blank
        else [phoneToId(p) for p in phonemes]
    )  # +1 to shift the ids by 1 as 0 is blank
    return PhonemeSeq(phoneme_ids, phonemes)


def decode_predicted_phoneme_ids(ids: list[int], zero_is_blank=True) -> str:
    return " ".join(
        [
            PHONE_DEF_SIL[(i - 1) if zero_is_blank else i]
            for i in ids
            if i > (0 if zero_is_blank else -1)
        ]
    )
