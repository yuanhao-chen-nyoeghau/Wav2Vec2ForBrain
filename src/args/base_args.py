from pydantic import BaseModel, Field
from typing import Literal


class BaseArgsModel(BaseModel):
    batch_size: int = Field(32, description="Batch size for training and validation")
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: Literal["adam", "sgd"] = "adam"
    loss_function: str = "mse"
    experiment_name: str = "experiment_1"
    experiment_type: str = Field("wav2vec")
