from typing import Literal, NamedTuple, cast

from git import Optional
import numpy as np
import torch
from src.datasets.batch_types import SampleBatch
from src.datasets.brain2text_w_phonemes import PHONE_DEF_SIL
from src.datasets.base_dataset import BaseDataset
from pydantic import BaseModel
import os
import soundfile
from src.args.yaml_config import YamlConfigModel


class Phoneme(NamedTuple):
    start: int
    end: int
    id: int


class TimitSample(NamedTuple):
    input: torch.Tensor
    target_id: int
    target_start: int
    target_end: int


class RawSample(NamedTuple):
    phonemes: list[Phoneme]
    transcript: str
    audio: np.ndarray


class TimitAudioDatasetArgsModel(BaseModel):
    limit_samples: Optional[int] = None


class TimitSampleBatch(SampleBatch):
    class_weights: torch.Tensor


class TimitAudioDataset(BaseDataset):
    def __init__(
        self,
        config: TimitAudioDatasetArgsModel,
        yaml_config: YamlConfigModel,
        split: Literal["train", "val", "test"],
    ):
        self.config = config
        splits_dir = yaml_config.timit_dataset_splits_dir
        partition = "TRAIN" if split == "train" else "TEST"
        data_folder = splits_dir + "/data/" + partition
        self.data: list[TimitSample] = []
        sample_folders = []
        for folder in os.listdir(data_folder):
            complete_path = data_folder + "/" + folder
            for sample_folder in os.listdir(complete_path):
                complete_sample_path = complete_path + "/" + sample_folder
                sample_folders.append(complete_sample_path)

        for folder in sample_folders:
            for sample in self._extractSampleDataFromFolder(folder):
                self.data += self._extract_ds_samples_from_raw_sample(sample)

        self.class_weights = torch.tensor(
            self._calculate_target_weights(), dtype=torch.float32
        )

    def __getitem__(self, index: int) -> TimitSample:
        sample = self.data[index]
        return sample

    def get_collate_fn(self):
        def _collate(samples: list[TimitSample]) -> SampleBatch:
            inputs = torch.stack([sample.input for sample in samples])
            targets = torch.tensor(
                [sample.target_id for sample in samples], dtype=torch.long
            )
            batch = TimitSampleBatch(inputs, targets)
            batch.class_weights = self.class_weights
            return batch

        return _collate

    def __len__(self):
        return (
            len(self.data)
            if self.config.limit_samples == None
            else min(self.config.limit_samples, len(self.data))
        )

    def _extractSampleDataFromFolder(self, folder: str) -> list[RawSample]:
        sampleNames = [
            fileName.split(".")[0]
            for fileName in os.listdir(folder)
            if fileName.split(".")[-1] == "TXT"
        ]
        samples: list[RawSample] = []
        for sampleName in sampleNames:
            phonemeFile = folder + "/" + sampleName + ".PHN"
            transcriptFile = folder + "/" + sampleName + ".TXT"
            audioFile = folder + "/" + sampleName + ".WAV"
            sample_dict = RawSample(
                self._readPhonemes(phonemeFile),
                self._readTranscript(transcriptFile),
                self._readAudio(audioFile),
            )
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

    def _extract_ds_samples_from_raw_sample(
        self, sample: RawSample
    ) -> list[TimitSample]:
        window_size = 400
        stride = window_size // 2
        result = []
        for p in sample.phonemes:
            # TODO: adapt after hist

            if p.id == -1:
                continue
            ph_len = p.end - p.start
            if ph_len < window_size:
                # If the phoneme is shorter than the window size, we pad it to the window size
                padding = window_size - ph_len
                audio_start = p.start - padding // 2
                audio_end = p.end + padding - padding // 2
                if audio_start < 0:
                    audio_start = 0
                    audio_end = window_size
                elif audio_end > len(sample.audio):
                    audio_end = len(sample.audio)
                    audio_start = audio_end - window_size
                audio_window = sample.audio[audio_start:audio_end]
                result.append(
                    TimitSample(
                        input=torch.tensor(audio_window, dtype=torch.float32),
                        target_id=p.id,
                        target_start=p.start,
                        target_end=p.end,
                    )
                )
            else:
                # If the phoneme is longer than the window size, we generate multiple samples from it based on stride
                audio_window = sample.audio[p.start : p.end]
                for i in range(0, len(audio_window) - window_size + 1, stride):
                    result.append(
                        TimitSample(
                            input=torch.tensor(
                                audio_window[i : i + window_size], dtype=torch.float32
                            ),
                            target_id=p.id,
                            target_start=p.start,
                            target_end=p.end,
                        )
                    )
        return result

    def _calculate_target_weights(self):
        counts = [0] * len(PHONE_DEF_SIL)
        for sample in self.data:
            counts[sample.target_id] += 1
        s = sum(counts)
        weights = np.array([s / c for c in counts])
        return weights / np.median(weights)
