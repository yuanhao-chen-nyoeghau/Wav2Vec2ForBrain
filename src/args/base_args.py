from pydantic import BaseModel, Field
from typing import Literal, Optional


class B2TDatasetArgsModel(BaseModel):
    preprocessing: Literal[
        "competition_recommended",
        "seperate_zscoring",
        "only_tx_unnormalized",
        "only_tx_zscored",
        "only_spikepow_unnormalized",
        "only_spikepow_zscored",
        "seperate_zscoring_2channels",
        "seperate_zscoring_4channels",
    ] = "seperate_zscoring_4channels"
    competition_mode: bool = False
    limit_samples: Optional[int] = Field(None, description="Limit number of samples")
    sample_rate: int = 50


class CTCTextDatasetArgsModel(BaseModel):
    limit_samples: Optional[int] = Field(None, description="Limit number of samples")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    avg_num_blank_after_char: int = 6
    insert_wrong_char_prob: float = 0.05
    remove_char_prob: float = 0.05
    noise_mean: float = -16
    noise_std: float = 2
    correct_as_second_prob = 0.2
    random_second_id_in_blank_prob: float = 0.1


class BaseExperimentArgsModel(BaseModel):
    batch_size: int = Field(16, description="Batch size for training and validation")
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: Literal["adam", "sgd"] = "adam"
    loss_function: Literal["ctc", "contrastive_loss"] = "ctc"
    ctc_loss_reduction: Literal["sum", "mean"] = "mean"
    experiment_name: str = "experiment_1"
    experiment_type: Literal[
        "b2t_wav2vec_sharedaggregation",
        "b2t_wav2vec_cnn",
        "audio_wav2vec2",
        "b2t_audio_wav2vec",
        "b2t_wav2vec_resnet",
        "b2t_wav2vec_pretraining",
        "b2t_wav2vec_custom_encoder",
        "onehot_index",
        "b2t_cnn",
        "b2t_gru",
        "mvts_transformer"
        "b2t_mamba",
        "ctc_lm",
    ] = Field("b2t_wav2vec_sharedaggregation")
    log_every_n_batches: int = 10
    scheduler: Literal["step"] = "step"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    return_best_model: bool = True
    best_model_metric: str = Field(
        "loss",
        description='The metric by which to measure the models performance. Can be "loss" for using the applied loss or any metric that is returned by the model',
    )
    minimize_best_model_metric: bool = Field(
        True,
        description="Specify if best_model_metric should be minimized or maximized",
    )
    use_wandb: bool = False
    from_checkpoint: Optional[str] = Field(
        None, description="(optional) Path to model checkpoint"
    )
    only_test: bool = Field(False, description="Only run test, skip training")
    predict_on_train: bool = Field(
        False, description="Run prediction on train set after model training"
    )
    remove_punctuation: bool = True
    tokenizer: Literal["wav2vec_pretrained", "ours"] = "wav2vec_pretrained"
    tokenizer_checkpoint: Literal["facebook/wav2vec2-base-100h", None] = (
        "facebook/wav2vec2-base-100h"
    )
    gradient_clipping: Optional[float] = None
    weight_decay: float = 0.0
    visualize_predictions_n_batches: int = 1


class B2TArgsModel(BaseExperimentArgsModel, B2TDatasetArgsModel):
    pass
