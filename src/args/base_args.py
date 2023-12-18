from pydantic import BaseModel, Field
from typing import Literal


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
    window_size: int = 20


class BaseExperimentArgsModel(B2TDatasetArgsModel):
    batch_size: int = Field(32, description="Batch size for training and validation")
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: Literal["adam", "sgd"] = "adam"
    loss_function: str = "mse"
    experiment_name: str = "experiment_1"
    experiment_type: str = Field("wav2vec")
    log_every_n_batches: int = 1000
    scheduler: Literal["step"] = "step"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    return_best_model: bool = True
    use_wandb: bool = False
