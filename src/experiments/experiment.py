from abc import ABC, abstractmethod, abstractclassmethod, ABCMeta
from torch.utils.data import Dataset
from pydantic import BaseModel
from src.datasets.brain2text import Brain2TextDataset
from src.args.base_args import BaseExperimentArgsModel
from torch.utils.data import default_collate
from src.model.b2tmodel import B2TModel, ModelOutput
from typing import Literal, Type, cast, Any
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
from src.train.history import TrainHistory
import sys

Optimizers: dict[str, Type[Optimizer]] = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}


class Experiment(metaclass=ABCMeta):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.base_config = BaseExperimentArgsModel(**config)
        self.yaml_config = yamlConfig

        self.tokenizer = self._create_tokenizer()

        self.dataloader_train = self._create_dataloader(split="train")
        self.dataloader_val = self._create_dataloader(split="val")
        self.dataloader_test = self._create_dataloader(split="test")

        self.checkpoint_history = None

        self.results_dir = os.path.join(
            yamlConfig.cache_dir,
            "experiment_results",
            self.get_name(),
            f"{datetime.now():%Y-%m-%d_%H#%M#%S}",
        )
        os.makedirs(self.results_dir, exist_ok=True)
        with open(os.path.join(self.results_dir, "config.json"), "w") as f:
            config_copy = dict(config)
            config_copy["repro_cmd"] = "python " + " ".join(sys.argv)
            json.dump(config_copy, f, indent=5)
        self.model = self._create_model().cuda()
        self.checkpoint_history = None
        if not self.base_config.from_checkpoint is None:
            print(f"loading model from checkpoint {self.base_config.from_checkpoint}")
            self.model.load_state_dict(
                torch.load(self.base_config.from_checkpoint, map_location="cuda"),
                strict=False,
            )
            history_path = os.path.join(
                os.path.dirname(self.base_config.from_checkpoint), "history.json"
            )
            if os.path.exists(history_path):
                print("Attempting to load history from checkpoint")
                try:
                    self.checkpoint_history = TrainHistory.from_json(history_path)
                except:
                    print("Failed to load history from checkpoint")

            print("")

    def run(self):
        from src.train.train_loop import Trainer

        if self.base_config.use_wandb:
            wandb.login(key=self.yaml_config.wandb_api_key, relogin=True)

        trainer = Trainer(self)
        wandb.init(
            project=self.yaml_config.wandb_project_name,
            entity=self.yaml_config.wandb_entity,
            config=self.base_config.dict(),
            name=self.base_config.experiment_name,
            dir=self.yaml_config.cache_dir,
            save_code=True,
            mode="online" if self.base_config.use_wandb else "disabled",
        )
        if wandb.run is None:
            raise Exception("wandb init failed. wandb.run is None")
        with wandb.run:
            wandb.watch(trainer.model)
            model_for_testing: B2TModel

            if not self.base_config.only_test:
                trained_model, history = trainer.train()
                torch.save(
                    trained_model.state_dict(),
                    os.path.join(self.results_dir, "model.pt"),
                )
                with open(os.path.join(self.results_dir, "history.json"), "w") as f:
                    json.dump(history.to_dict(), f, indent=5)
                model_for_testing = trained_model

                self.plot_results(history)
            else:
                model_for_testing = self.model

            self.run_real_world_test(model_for_testing)

            print(f"Done. Saved results to {self.results_dir}")

    def plot_results(self, history: TrainHistory):
        history.plot(os.path.join(self.results_dir, "history.png"))

    def run_real_world_test(self, model: B2TModel):
        self._predict_and_store(model, self.dataloader_test, "test_prediction.json")
        if self.base_config.predict_on_train == True:
            self._predict_and_store(
                model, self.dataloader_train, "train_prediction.json"
            )

    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_collate_fn(self):
        return default_collate

    @abstractmethod
    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> Dataset:
        raise NotImplementedError("Implement _create_dataset in subclass")

    @abstractmethod
    def _create_tokenizer(self) -> PreTrainedTokenizer:
        pass

    @abstractmethod
    def _create_model(self) -> B2TModel:
        pass

    @abstractmethod
    def get_args_model() -> Type[BaseExperimentArgsModel]:
        raise NotImplementedError()

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        return DataLoader(
            self._create_dataset(split),
            batch_size=self.base_config.batch_size,
            shuffle=True,
            collate_fn=self.get_collate_fn(),
        )

    def _predict_and_store(
        self, model: B2TModel, dataloader: DataLoader, out_file: str
    ):
        prediction = self._predict(model, dataloader)
        with open(os.path.join(self.results_dir, out_file), "w") as f:
            json.dump(prediction, f, indent=5)

    def _predict(self, model: B2TModel, dataloader: DataLoader):
        result = []
        for i, data in enumerate(dataloader):
            inputs, labels = data

            with torch.no_grad():
                outputs = model.forward(inputs.cuda())

                predicted_ids = outputs.logits.argmax(dim=-1).cpu().numpy()
                predicted = self.tokenizer.batch_decode(predicted_ids)
                result.append(
                    {
                        "predicted": predicted,
                        "label": self.tokenizer.batch_decode(
                            labels.cpu().numpy(), group_tokens=False
                        ),
                    }
                )
            print(
                f"Running predictions on test. Batch {i + 1}/{len(self.dataloader_test)}\r",
                end="",
            )
        return result

    def _get_optimizer_cls(self) -> Type[Optimizer]:
        if self.base_config.optimizer not in Optimizers:
            raise ValueError(
                f"Optimizer {self.base_config.optimizer} not implemented. "
                f"Choose from {Optimizers.keys()} or implement your own."
            )
        return Optimizers[self.base_config.optimizer]

    def create_optimizer(self) -> Optimizer:
        optim_cls: Any = self._get_optimizer_cls()
        return optim_cls(self.model.parameters(), lr=self.base_config.learning_rate)
