from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from torch.nn.functional import interpolate

Area = Literal["44", "6v"]


def resample_sample(
    sample: torch.Tensor, target_sample_rate: int, orig_sample_rate: int
):
    # Original window size is 20
    if target_sample_rate == orig_sample_rate:
        return sample

    # size: 187, 256

    scale_factor = target_sample_rate // orig_sample_rate
    interpolated = interpolate(
        sample.unsqueeze(0).transpose(-1, -2),
        scale_factor=scale_factor,
        mode="linear",
    ).transpose(-1, -2)
    return interpolated.squeeze(0)


def preprocess_competition_recommended(
    data_file: dict,
    block_index_ranges: list[np.ndarray[Any, np.dtype[np.int32]]],
    area: Area,
):
    n_trials = data_file["sentenceText"].shape[0]
    input_features = []
    transcriptions = []
    preprocessed_features = []
    preprocessed_transcriptions = []
    # collect area 6v tx1 and spikePow features
    for i in range(n_trials):
        if area == "44":
            tx_features = data_file["tx1"][0, i][:, 128:]
            spike_features = data_file["spikePow"][0, i][:, 128:]
        else:
            tx_features = data_file["tx1"][0, i][:, :128]
            spike_features = data_file["spikePow"][0, i][:, :128]
        # get time series of TX and spike power for this trial
        # first 128 columns = area 6v only
        features = np.concatenate(
            [
                tx_features,
                spike_features,
            ],
            axis=1,
        )
        sentence = data_file["sentenceText"][i].strip()
        input_features.append(features)
        transcriptions.append(sentence)

    for block_index_range in block_index_ranges:
        block_features = np.concatenate(
            input_features[block_index_range[0] : (block_index_range[-1] + 1)], axis=0
        )
        block_feats_mean = np.mean(block_features, axis=0, keepdims=True)
        block_feats_std = np.std(block_features, axis=0, keepdims=True)
        for i in block_index_range:
            preprocessed_features.append(
                (input_features[i] - block_feats_mean) / (block_feats_std + 1e-8)
            )
            preprocessed_transcriptions.append(transcriptions[i])

    return preprocessed_features, preprocessed_transcriptions


def _fn_preprocess_single_feature(
    feature: Literal["tx1", "spikePow"], apply_zscore: bool
):
    def preprocess_single_feature(
        data_file: dict,
        block_index_ranges: list[np.ndarray[Any, np.dtype[np.int32]]],
        area: Area,
    ):
        n_trials = data_file["sentenceText"].shape[0]
        features = []
        transcriptions = []
        preprocessed_features = []
        preprocessed_transcriptions = []

        for i in range(n_trials):
            if area == "44":
                trial_features = data_file[feature][0, i][:, 128:]
            else:
                trial_features = data_file[feature][0, i][:, :128]
            sentence = data_file["sentenceText"][i].strip()
            features.append(trial_features)
            transcriptions.append(sentence)

        for block_index_range in block_index_ranges:
            # z-score features
            block_features = np.concatenate(
                features[block_index_range[0] : (block_index_range[-1] + 1)],
                axis=0,
            )
            block_feats_mean = np.mean(block_features, axis=0, keepdims=True)
            block_feats_std = np.std(block_features, axis=0, keepdims=True)
            for i in block_index_range:
                preprocessed_features.append(
                    ((features[i] - block_feats_mean) / (block_feats_std + 1e-8))
                    if apply_zscore
                    else features[i]
                )
                preprocessed_transcriptions.append(transcriptions[i])

        return preprocessed_features, preprocessed_transcriptions

    return preprocess_single_feature


preprocess_only_tx_unnormalized = _fn_preprocess_single_feature(
    apply_zscore=False, feature="tx1"
)
preprocess_only_tx_zscored = _fn_preprocess_single_feature(
    apply_zscore=True, feature="tx1"
)
preprocess_only_spikepow_unnormalized = _fn_preprocess_single_feature(
    apply_zscore=False, feature="spikePow"
)
preprocess_only_spikepow_zscored = _fn_preprocess_single_feature(
    apply_zscore=True, feature="spikePow"
)


def preprocess_seperate_zscoring(
    data_file: dict,
    block_index_ranges: list[np.ndarray[Any, np.dtype[np.int32]]],
    area: Area,
):
    tx_features, transcriptions = preprocess_only_tx_zscored(
        data_file, block_index_ranges, area
    )
    spike_features, _ = preprocess_only_spikepow_zscored(
        data_file, block_index_ranges, area
    )
    assert len(tx_features) == len(
        spike_features
    ), "Length of tx and spike features must be equal."
    features = [
        np.concatenate(
            [
                tx_features[i],
                spike_features[i],
            ],
            axis=1,
        )
        for i in range(len(tx_features))
    ]

    return features, transcriptions


def preprocess_seperate_zscoring_2channels(
    data_file: dict,
    block_index_ranges: list[np.ndarray[Any, np.dtype[np.int32]]],
    area: Area,
):
    tx_features, transcriptions = preprocess_only_tx_zscored(
        data_file, block_index_ranges, area
    )
    spike_features, _ = preprocess_only_spikepow_zscored(
        data_file, block_index_ranges, area
    )
    assert len(tx_features) == len(
        spike_features
    ), "Length of tx and spike features must be equal."

    features = [
        np.stack(
            [
                tx_features[i],
                spike_features[i],
            ],
            axis=0,
        )
        for i in range(len(tx_features))
    ]

    return features, transcriptions


def preprocess_seperate_zscoring_4channels(
    data_file: dict,
    block_index_ranges: list[np.ndarray[Any, np.dtype[np.int32]]],
    area: Area,
):
    tx_features, transcriptions = preprocess_only_tx_zscored(
        data_file, block_index_ranges, area
    )
    spike_features, _ = preprocess_only_spikepow_zscored(
        data_file, block_index_ranges, area
    )

    features = [
        np.stack(
            [
                tx_features[i][:, :64],
                tx_features[i][:, 64:],
                spike_features[i][:, :64],
                spike_features[i][:, 64:],
            ],
            axis=0,
        )
        for i in range(len(tx_features))
    ]

    return features, transcriptions
