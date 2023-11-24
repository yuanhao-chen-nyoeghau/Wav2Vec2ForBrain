from pydantic import BaseModel, Field


class BaseArgsModel(BaseModel):
    batch_size: int = Field(32, description="Batch size for training and validation")
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "mse"
    experiment_name: str = "experiment_1"
    experiment_type: str = Field("wav2vec")
    log_every_n_batches: int = 1000
    scheduler: str = "step"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
