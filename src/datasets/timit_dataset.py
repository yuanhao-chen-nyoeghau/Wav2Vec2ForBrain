import re
from typing import Literal, NamedTuple, override

import numpy as np
import torch
from src.datasets.batch_types import SampleBatch
from src.datasets.brain2text_w_phonemes import PHONE_DEF_SIL
from src.datasets.audio_with_phonemes import AudioWPhonemesDatasetArgsModel
from src.datasets.base_dataset import BaseDataset, Sample
from pydantic import BaseModel
import os
import soundfile
from torch.nn.functional import pad


from args.yaml_config import YamlConfigModel


class Phoneme(NamedTuple):
    start: int
    end: int
    id: int


class TimitSample(NamedTuple):
    target: list[Phoneme]
    transcript: str
    input: torch.Tensor


class TimitSampleBatch(SampleBatch):
    transcripts: list[str]
    input_lens: torch.Tensor
    phonemes: list[list[Phoneme]]


class TimitAudioDataset(BaseDataset):
    def __init__(
        self,
        config: AudioWPhonemesDatasetArgsModel,
        yaml_config: YamlConfigModel,
        split: Literal["train", "val", "test"],
    ):
        self.config = config
        splits_dir = yaml_config.dataset_splits_dir
        partition = "TRAIN" if split == "train" else "TEST"
        data_folder = splits_dir + "/data/" + partition
        self.data: list[dict] = []
        sample_folders = []
        for folder in os.listdir(data_folder):
            complete_path = data_folder + "/" + folder
            for sample_folder in os.listdir(complete_path):
                complete_sample_path = complete_path + "/" + sample_folder
                sample_folders.append(complete_sample_path)

        for folder in sample_folders:
            for sample in self._extractSampleDataFromFolder(folder):
                self.data.append(sample)

    def __getitem__(self, index) -> TimitSample:
        row = self.data[index]
        transcription = row["transcript"].upper()
        if self.config.remove_punctuation:
            chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'
            transcription = re.sub(chars_to_ignore_regex, "", transcription)
        sample = TimitSample(
            target=row["phonemes"],
            transcript=transcription,
            input=torch.tensor(row["audio"], dtype=torch.float32),
        )
        return sample

    def get_collate_fn(self):
        def _collate(samples: list[TimitSample]):
            max_audio_len = max([audio.size(0) for _, _, audio in samples])

            padded_audio = [
                pad(
                    audio,
                    (0, max_audio_len - audio.size(0)),
                    mode="constant",
                    value=0,
                )
                for _, _, audio in samples
            ]

            target_tensors = []
            for phonemes, _, audio in samples:
                target_tensor = torch.full((audio.size(0),), -1, dtype=torch.long)
                for phoneme in phonemes:
                    target_tensor[phoneme.start : phoneme.end] = phoneme.id
                target_tensors.append(target_tensor)

            max_target_len = max(
                [target_tensor.size(0) for target_tensor in target_tensors]
            )
            padded_targets = [
                pad(
                    target_tensor,
                    (0, max_target_len - target_tensor.size(0)),
                    mode="constant",
                    value=-1,
                )
                for target_tensor in target_tensors
            ]

            batch = TimitSampleBatch(
                input=torch.stack(padded_audio), target=torch.stack(padded_targets)
            )
            batch.transcripts = [transcript for _, transcript, _ in samples]
            batch.input_lens = torch.tensor([audio.size(0) for _, _, audio in samples])
            batch.phonemes = [phonemes for phonemes, _, _ in samples]

            return batch

        return _collate

    def __len__(self):
        return len(self.data)

    def _extractSampleDataFromFolder(self, folder: str) -> list[dict]:
        sampleNames = [
            fileName.split(".")[0]
            for fileName in os.listdir(folder)
            if fileName.split(".")[-1] == "TXT"
        ]
        samples = []
        for sampleName in sampleNames:
            phonemeFile = folder + "/" + sampleName + ".PHN"
            transcriptFile = folder + "/" + sampleName + ".TXT"
            audioFile = folder + "/" + sampleName + ".WAV"
            sample_dict = {
                "phonemes": self._readPhonemes(phonemeFile),
                "transcript": self._readTranscript(transcriptFile),
                "audio": self._readAudio(audioFile),
            }
            samples.append(sample_dict)

        return samples

    def _readAudio(self, filePath: str) -> np.ndarray:
        # Frequency should be 16 kHz as used in the pretraining dataset of wav2vec
        # V. Panayotov, G. Chen, D. Povey, and S. Khudanpur. Librispeech: an asr corpus based on public
        # domain audio books. In Proc. of ICASSP, pages 5206–5210. IEEE, 2015.
        wav = soundfile.read(filePath)
        return np.array(wav[0])

    def _readTranscript(self, filePath: str) -> str:
        with open(filePath, "r") as f:
            transcript = " ".join(f.read().split(" ")[2:])
        return transcript

    def _readPhonemes(self, filePath: str) -> list[Phoneme]:

        def phoneToId(p):
            return PHONE_DEF_SIL.index(p)

        phonemes = []
        with open(filePath, "r") as f:
            for line in f.readlines():
                line_content = line.split(" ")
                # We remove trailing newline and closure stop tag from phonemes (see Phoneme docs)
                raw_phoneme = (
                    line_content[2].replace("\n", "").replace("cl", "").upper()
                )
                # There are special phonemes for different pauses etc
                # Wav2Vec only knows one pause
                # TODO: correct mapping (check phoneme atlas)
                silence_phonemes = ["PAU", "EPI", "H#"]
                for silence_phoneme in silence_phonemes:
                    raw_phoneme = raw_phoneme.replace(silence_phoneme, "SIL")

                try:
                    phoneme_id = phoneToId(raw_phoneme)
                except ValueError:
                    # Phoneme is not known
                    phoneme_id = -1

                phoneme = Phoneme(
                    start=int(line_content[0]),
                    end=int(line_content[1]),
                    id=phoneme_id,
                )
                phonemes.append(phoneme)
        return phonemes
