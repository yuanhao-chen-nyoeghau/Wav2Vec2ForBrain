from src.args.base_args import B2TDatasetArgsModel
from src.args.wav2vec_args import AudioWav2VecArgsModel


class B2TAudioWav2VecArgsModel(AudioWav2VecArgsModel):
    hidden_nodes: int = 16
    mean_reduction_model: bool = False
    feature_extraction_per_feature: bool = True


class B2TAudioDatasetArgsModel(B2TDatasetArgsModel):
    smoothing_window: int = 50
    audio_smoothing_window: int = 5
    audio_frequency: int = 16000
    frequency_coefficient: float = 50
    mean_reduction_data: bool = False
    remove_punctuation: bool = True
