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
    ] = "seperate_zscoring"
    competition_mode: bool = False
    limit_samples: Optional[int] = Field(None, description="Limit number of samples")
    window_size: int = 20


class BaseExperimentArgsModel(BaseModel):
    batch_size: int = Field(16, description="Batch size for training and validation")
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: Literal["adam", "sgd"] = "adam"
    loss_function: Literal["ctc"] = "ctc"
    experiment_name: str = "experiment_1"
    experiment_type: Literal[
        "b2t_wav2vec_fc", "b2t_wav2vec_cnn", "audio_wav2vec2"
    ] = Field("b2t_wav2vec")
    log_every_n_batches: int = 10
    scheduler: Literal["step"] = "step"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    return_best_model: bool = True
    use_wandb: bool = False
    from_checkpoint: Optional[str] = Field(
        None, description="(optional) Path to model checkpoint"
    )
    only_test: bool = Field(False, description="Only run test, skip training")
    predict_on_train: bool = Field(
        False, description="Run prediction on train set after model training"
    )
