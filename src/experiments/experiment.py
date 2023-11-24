from abc import ABC, abstractmethod, abstractclassmethod
from torch.utils.data import Dataset
from argparse import ArgumentParser
from pydantic import BaseModel
from src.args.base_args import BaseArgsModel
from torch.utils.data import default_collate
from src.model.b2tmodel import B2TModel
from typing import Literal
from torch.nn.modules.loss import _Loss
from src.args.yaml_config import YamlConfigModel
import wandb


class Experiment(ABC):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = BaseArgsModel(**config)
        self.yaml_config = yamlConfig

    def run(self):
        from src.train.train_loop import Trainer

        if self.config.use_wandb:
            wandb.login(key=self.yaml_config.wandb_api_key, relogin=True)

        trainer = Trainer(self)
        with wandb.init(
            project=self.yaml_config.wandb_project_name,
            entity=self.yaml_config.wandb_entity,
            config=self.config.dict(),
            name=self.config.experiment_name,
            dir=self.yaml_config.cache_dir,
            save_code=True,
            mode="online" if self.config.use_wandb else "disabled",
        ):
            trained_model, history = trainer.train()

    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_collate_fn(self):
        return default_collate

    def get_dataset(self, split=Literal["train", "val", "test"]) -> Dataset:
        pass

    def get_model(self) -> B2TModel:
        pass

    def get_loss_function(self) -> _Loss:
        pass

    @abstractclassmethod
    def get_args_model() -> BaseModel:
        pass
