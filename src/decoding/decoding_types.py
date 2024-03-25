from typing import NamedTuple, Optional


class LLMOutput(NamedTuple):
    cer: float
    wer: float
    decoded_transcripts: list[str]
    confidences: Optional[list[float]]
    target_transcripts: list[str]
