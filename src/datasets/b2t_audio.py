from typing import Any, Literal, Callable
from torch.utils.data import Dataset
import os
from scipy.io import loadmat
from pathlib import Path
import torch
import numpy as np
from src.args.yaml_config import YamlConfigModel
from src.args.base_args import B2TDatasetArgsModel
from src.datasets.preprocessing import (
    preprocess_competition_recommended,
    preprocess_seperate_zscoring,
)
from tqdm import tqdm

PreprocessingFunctions: dict[
    str,
    Callable[[dict, list[np.ndarray[Any, np.dtype[np.int32]]]], tuple[list, list[str]]],
] = {
    "competition_recommended": preprocess_competition_recommended,
    "seperate_zscoring": preprocess_seperate_zscoring,
}


def construct_audio_sig(
    amplitudes: np.ndarray, frequencies: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    time_stamps = np.array([])
    signal = np.array([])
    pos = True

    # Input amplitudes and frequencies are spaced in 20 ms windows
    orig_spacing = 20.0

    for i in range(len(amplitudes)):
        freq = frequencies[i]
        amp = amplitudes[i]
        if freq != 0:
            time_stamp_offset = orig_spacing / freq
            for j in range(freq):
                time_stamp = orig_spacing * i + j * time_stamp_offset
                signal_point = (amp if pos else -amp) * 10
                pos = not pos
                time_stamps = np.append(time_stamps, time_stamp)
                signal = np.append(signal, signal_point)

    # Time stamp unit is in ms, we convert to s
    time_stamps = time_stamps / 1000
    return time_stamps, signal


def sample_raw_signal(sample_rate, x, y):
    start_time = x[0]
    end_time = x[-1]
    target_freq_s = 1.0 / sample_rate
    new_x = np.arange(start_time, end_time + target_freq_s, target_freq_s)
    new_y = np.interp(new_x, x, y)
    return new_x, new_y


def b2t_audio_transformation(spike_pows, spike_counts, smoothing_window, sampling_rate):
    # Rolling mean over features
    kernel = torch.ones(smoothing_window) / smoothing_window
    kernel = kernel.cuda()

    spike_powers = torch.nn.functional.conv1d(
        spike_pows.view(1, 1, -1),
        kernel.view(1, 1, -1),
        padding=smoothing_window // 2,
    ).view(-1)[1:]
    spike_counts = torch.nn.functional.conv1d(
        spike_counts.view(1, 1, -1),
        kernel.view(1, 1, -1),
        padding=smoothing_window // 2,
    ).view(-1)[1:]

    frequencies = spike_counts.cpu().numpy()
    min_freq = frequencies.min()
    frequencies = frequencies + abs(min_freq)
    frequencies = frequencies * 100
    frequencies = frequencies.astype(np.int16)

    amplitudes = spike_powers.cpu().numpy()
    min_freq = amplitudes.min()
    amplitudes = amplitudes + abs(min_freq)

    x, y = construct_audio_sig(amplitudes=amplitudes, frequencies=frequencies)
    x, y = sample_raw_signal(sampling_rate, x, y)
    return y


class B2TAudioDataset(Dataset):
    def __init__(
        self,
        config: B2TDatasetArgsModel,
        yaml_config: YamlConfigModel,
        split: Literal["train", "val", "test"] = "train",
        smoothing_window: int = 50,
        sampling_rate: int = 16000,
        mean_reduction: bool = True,
    ) -> None:
        super().__init__()
        self.config = config

        if split == "val":
            data_path = Path(yaml_config.dataset_splits_dir) / "test"
        elif split == "test" and config.competition_mode:
            data_path = Path(yaml_config.dataset_splits_dir) / "competitionHoldOut"
        else:
            data_path = Path(yaml_config.dataset_splits_dir) / "train"

        if not os.path.exists(data_path):
            raise Exception(f"{data_path} does not exist.")

        data_files = [
            loadmat(data_path / fileName) for fileName in os.listdir(data_path)
        ]

        print("Loaded raw data")

        self.transcriptions: list[str] = []
        brain_data_samples: list[torch.Tensor] = []
        preprocess = PreprocessingFunctions[config.preprocessing]

        print("Preprocessing brain data and sentence data")
        for data_file in tqdm(data_files):
            # block-wise feature normalization
            blockNums = np.squeeze(data_file["blockIdx"])
            blockList = np.unique(blockNums)

            if split == "test" and not config.competition_mode:
                blockList = [blockList[0]]
            if split == "train" and not config.competition_mode:
                blockList = blockList[1:]

            blocks = []
            for b in range(len(blockList)):
                sentIdx = np.argwhere(blockNums == blockList[b])
                sentIdx = sentIdx[:, 0].astype(np.int32)
                blocks.append(sentIdx)

            input_features, transcriptions = preprocess(data_file, blocks)

            for dataSample in input_features:
                brain_data_samples.append(torch.tensor(dataSample, dtype=torch.float32))
            for sentence in transcriptions:
                self.transcriptions.append(f"<s>{sentence.upper()}</s>")

        if self.config.limit_samples is not None:
            brain_data_samples = brain_data_samples[: config.limit_samples]
            self.transcriptions = self.transcriptions[: config.limit_samples]

        print("Converting to sound waves")
        self.soundwaves: list[torch.Tensor] = []
        for sample in tqdm(brain_data_samples):
            # TODO add no reduction behavior
            if mean_reduction:
                spike_powers = sample[:, :128].mean(-1).cuda()
                spike_counts = sample[:, 128:].mean(-1).cuda()
                y = b2t_audio_transformation(
                    spike_powers, spike_counts, smoothing_window, sampling_rate
                )
            else:
                sound_arrays = []
                neuron_count = 128
                for i in range(neuron_count):
                    spike_powers = sample[:, i].cuda()
                    spike_counts = sample[:, 128 + i].cuda()
                    neuron_y = b2t_audio_transformation(
                        spike_powers, spike_counts, smoothing_window, sampling_rate
                    )
                    sound_arrays.append(neuron_y)
                min_len = min([len(wave) for wave in sound_arrays])
                # Resampling sometimes results in different size arrays
                sound_arrays = [sound_array[:min_len] for sound_array in sound_arrays]
                y = np.stack(sound_arrays, axis=-1)
            y = torch.from_numpy(y).float()
            self.soundwaves.append(y)

        assert len(self.transcriptions) == len(
            self.soundwaves
        ), "Length of labels and data samples must be equal."

    def __len__(self):
        return (
            len(self.transcriptions)
            if self.config.limit_samples is None
            else self.config.limit_samples
        )

    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        return self.soundwaves[index], self.transcriptions[index]
