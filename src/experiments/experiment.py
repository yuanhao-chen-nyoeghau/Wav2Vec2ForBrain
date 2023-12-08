from abc import ABC, abstractmethod, abstractclassmethod
from torch.utils.data import Dataset
from pydantic import BaseModel
from src.datasets.brain2text import Brain2TextDataset
from src.args.base_args import BaseExperimentArgsModel
from torch.utils.data import default_collate
from src.model.b2tmodel import B2TModel
from typing import Literal
from torch.nn.modules.loss import _Loss
from src.args.yaml_config import YamlConfigModel
import wandb
from torch.utils.data import DataLoader
import torch
from transformers import PreTrainedTokenizer
import json
import os
from datetime import datetime
from torch.optim.optimizer import Optimizer

Optimizers: dict[str, Optimizer.__class__] = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}


class Experiment(ABC):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.base_config = BaseExperimentArgsModel(**config)
        self.yaml_config = yamlConfig

        self.tokenizer = self._create_tokenizer()

        self.dataloader_train = self._create_dataloader(split="train")
        self.dataloader_val = self._create_dataloader(split="val")
        self.dataloader_test = self._create_dataloader(split="test")

        self.results_dir = os.path.join(
            yamlConfig.cache_dir,
            "experiment_results",
            self.get_name(),
            f"{datetime.now():%Y-%m-%d_%H#%M#%S}",
        )
        os.makedirs(self.results_dir, exist_ok=True)
        with open(os.path.join(self.results_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=5)
        self.model = self._create_model().cuda()
        if not self.base_config.from_checkpoint is None:
            self.model.load_state_dict(
                torch.load(self.base_config.from_checkpoint, map_location="cuda")
            )

    def run(self):
        from src.train.train_loop import Trainer

        if self.base_config.use_wandb:
            wandb.login(key=self.yaml_config.wandb_api_key, relogin=True)

        trainer = Trainer(self)
        with wandb.init(
            project=self.yaml_config.wandb_project_name,
            entity=self.yaml_config.wandb_entity,
            config=self.base_config.dict(),
            name=self.base_config.experiment_name,
            dir=self.yaml_config.cache_dir,
            save_code=True,
            mode="online" if self.base_config.use_wandb else "disabled",
        ):
            wandb.watch(trainer.model)
            model_for_testing: B2TModel

            if not self.config.only_test:
                trained_model, history = trainer.train()
                torch.save(
                    trained_model.state_dict(),
                    os.path.join(self.results_dir, "model.pt"),
                )
                with open(os.path.join(self.results_dir, "history.json"), "w") as f:
                    json.dump(history.to_dict(), f, indent=5)
                model_for_testing = trained_model
            else:
                model_for_testing = self.model

            test_prediction = self._predict_on_test(model_for_testing)
            with open(os.path.join(self.results_dir, "test_prediction.json"), "w") as f:
                json.dump(test_prediction, f, indent=5)

            print(f"Done. Saved results to {self.results_dir}")

    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_collate_fn(self):
        return default_collate

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> Dataset:
        return Brain2TextDataset(
            config=self.base_config,
            yaml_config=self.yaml_config,
            split=split,
            tokenizer=self.tokenizer,
        )

    @abstractmethod
    def _create_tokenizer(self) -> PreTrainedTokenizer:
        pass

    @abstractmethod
    def _create_model(self) -> B2TModel:
        pass

    @abstractclassmethod
    def get_args_model() -> BaseExperimentArgsModel.__class__:
        pass

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        return DataLoader(
            self._create_dataset(split),
            batch_size=self.base_config.batch_size,
            shuffle=True,
            collate_fn=self.get_collate_fn(),
        )

    def _predict_on_test(self, model: B2TModel):
        result = []
        for i, data in enumerate(self.dataloader_test):
            inputs, labels = data

            with torch.no_grad():
                outputs = model.forward(inputs.cuda())

                predicted_ids = outputs.logits.argmax(dim=-1).cpu().numpy()
                predicted = self.tokenizer.batch_decode(predicted_ids)
                result.append(
                    {
                        "predicted": predicted,
                        "label": self.tokenizer.batch_decode(labels.cpu().numpy()),
                    }
                )
            print(
                f"Running predictions on test. Batch {i + 1}/{len(self.dataloader_test)}\r",
                end="",
            )
        return result

    def _get_optimizer_cls(self) -> Optimizer.__class__:
        if self.config.optimizer not in Optimizers:
            raise ValueError(
                f"Optimizer {self.config.optimizer} not implemented. "
                f"Choose from {Optimizers.keys()} or implement your own."
            )
        return Optimizers[self.base_config.optimizer]

    def create_optimizer(self) -> Optimizer:
        return self._get_optimizer_cls()(
            self.model.parameters(), lr=self.base_config.learning_rate
        )
