from typing import NamedTuple, Optional
import numpy as np


class LLMOutput(NamedTuple):
    cer: float
    wer: float
    decoded_transcripts: list[str]
    confidences: Optional[list[float]]
    target_transcripts: list[str]


def text_to_ascii(text: str):
    return [ord(char) for char in text]


def prepare_transcription_batch(transcriptions: list[str]):
    transcriptions = [
        t.replace("<s>", "").replace("</s>", "").lower() for t in transcriptions
    ]
    max_len = (
        max([len(t) for t in transcriptions]) + 1
    )  # make sure there is an end token/blank in each sequence
    paddedTranscripts = []
    for t in transcriptions:
        paddedTranscription = np.zeros([max_len]).astype(np.int32)
        paddedTranscription[0 : len(t)] = np.array(text_to_ascii(t))
        paddedTranscripts.append(paddedTranscription)
    return np.stack(paddedTranscripts, axis=0)
