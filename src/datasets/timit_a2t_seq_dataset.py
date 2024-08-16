import re
from typing import Literal, NamedTuple
import numpy as np
import torch
from src.datasets.batch_types import SampleBatch
from src.datasets.audio_with_phonemes_seq import AudioWPhonemesDatasetArgsModel
from src.datasets.base_dataset import BaseDataset
import os
import soundfile
from torch.nn.functional import pad
from src.args.yaml_config import YamlConfigModel
from transformers import PreTrainedTokenizer
from src.util.nn_helper import calc_seq_len


class TimitA2TSeqSample(NamedTuple):
    input: torch.Tensor
    transcript: str


class TimitA2TSeqSampleBatch(SampleBatch):
    transcripts: list[str]
    input_lens: torch.Tensor
    target_lens: torch.Tensor


class TimitA2TSeqDataset(BaseDataset):
    def __init__(
        self,
        config: AudioWPhonemesDatasetArgsModel,
        yaml_config: YamlConfigModel,
        split: Literal["train", "val", "test"],
        tokenizer: PreTrainedTokenizer,
    ):
        self.config = config
        splits_dir = yaml_config.timit_dataset_splits_dir
        partition = "TRAIN" if split == "train" else "TEST"
        data_folder = splits_dir + "/data/" + partition
        self.data: list[TimitA2TSeqSample] = []
        sample_folders = []
        self.tokenizer = tokenizer
        for folder in os.listdir(data_folder):
            complete_path = data_folder + "/" + folder
            for sample_folder in os.listdir(complete_path):
                complete_sample_path = complete_path + "/" + sample_folder
                sample_folders.append(complete_sample_path)

        for folder in sample_folders:
            for sample in self._extractSampleDataFromFolder(folder):
                self.data.append(sample)

    def __getitem__(self, index) -> TimitA2TSeqSample:
        return self.data[index]

    def get_collate_fn(self):
        def _collate(samples: list[TimitA2TSeqSample]):
            max_audio_len = max([audio.size(0) for audio, _ in samples])

            padded_audio = [
                pad(
                    audio,
                    (0, max_audio_len - audio.size(0)),
                    mode="constant",
                    value=0,
                )
                for audio, _ in samples
            ]

            batch_label_ids: torch.Tensor = self.tokenizer(
                [label for _, label in samples],
                padding="longest",
                return_tensors="pt",
            ).input_ids

            batch = TimitA2TSeqSampleBatch(
                input=torch.stack(padded_audio), target=batch_label_ids
            )
            batch.transcripts = [transcript for _, transcript in samples]
            batch.input_lens = torch.tensor([audio.size(0) for audio, _ in samples])
            batch.target_lens = torch.tensor(
                [calc_seq_len(label_ids) for label_ids in batch_label_ids]
            )
            return batch

        return _collate

    def __len__(self):
        return len(self.data)

    def _extractSampleDataFromFolder(self, folder: str) -> list[TimitA2TSeqSample]:
        sampleNames = [
            fileName.split(".")[0]
            for fileName in os.listdir(folder)
            if fileName.split(".")[-1] == "TXT"
        ]
        samples: list[TimitA2TSeqSample] = []
        for sampleName in sampleNames:
            transcriptFile = folder + "/" + sampleName + ".TXT"
            audioFile = folder + "/" + sampleName + ".WAV"

            transcript = self._readTranscript(transcriptFile)
            audio = self._readAudio(audioFile)

            sample = TimitA2TSeqSample(
                torch.tensor(audio, dtype=torch.float32),
                transcript,
            )
            samples.append(sample)

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

        transcript = transcript.upper()
        if self.config.remove_punctuation:
            chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'
            transcript = re.sub(chars_to_ignore_regex, "", transcript)
        return transcript
