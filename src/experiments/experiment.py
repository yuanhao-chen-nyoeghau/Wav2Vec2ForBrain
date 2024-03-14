from abc import abstractmethod, ABCMeta
from git import Optional
from src.datasets.base_dataset import BaseDataset, SampleBatch
from src.args.base_args import BaseExperimentArgsModel
from src.model.b2tmodel import B2TModel, ModelOutput
from typing import Callable, Literal, Type, cast, Any, NamedTuple
from src.args.yaml_config import YamlConfigModel
import wandb
from torch.utils.data import DataLoader
import torch
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


class DecodedPredictionBatch(NamedTuple):
    predictions: list[str]
    targets: list[str]


class Experiment(metaclass=ABCMeta):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.base_config = BaseExperimentArgsModel(**config)
        self.yaml_config = yamlConfig

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
        self._predict_and_store(model, self.dataloader_test, "test")
        if self.base_config.predict_on_train == True:
            self._predict_and_store(model, self.dataloader_train, "train")

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        raise NotImplementedError("Implement _create_dataset in subclass")

    @abstractmethod
    def _create_model(self) -> B2TModel:
        pass

    @abstractmethod
    def get_args_model() -> Type[BaseExperimentArgsModel]:
        raise NotImplementedError()

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        ds = self._create_dataset(split)

        return DataLoader(
            ds,
            batch_size=self.base_config.batch_size,
            shuffle=True,
            collate_fn=ds.get_collate_fn(),
        )

    def _predict_and_store(
        self, model: B2TModel, dataloader: DataLoader, out_file_prefix: str
    ):
        def handle_evaluation_prediction_batch(
            batch_id: int,
            batch: SampleBatch,
            outputs: ModelOutput,
        ):
            if batch_id >= self.base_config.visualize_predictions_n_batches:
                return
            out_dir = os.path.join(self.results_dir, f"{out_file_prefix}_predictions")
            os.makedirs(out_dir, exist_ok=True)
            print(
                f"\nVisualizing prediction batch {batch_id+1}/{self.base_config.visualize_predictions_n_batches} for {out_file_prefix}..."
            )
            self.visualize_predictions(
                batch,
                outputs,
                os.path.join(out_dir, f"batch_{batch_id}.png"),
                batch_id,
            )

        prediction = self._predict(
            model, dataloader, handle_evaluation_prediction_batch
        )
        with open(
            os.path.join(self.results_dir, f"{out_file_prefix}_predictions.json"), "w"
        ) as f:
            json.dump(prediction, f, indent=5)

    @abstractmethod
    def decode_predictions(
        self, predictions: ModelOutput, sample: SampleBatch
    ) -> DecodedPredictionBatch:
        raise NotImplementedError("Implement decode_predictions in subclass")

    def _predict(
        self,
        model: B2TModel,
        dataloader: DataLoader,
        handle_prediction_batch: Optional[
            Callable[[int, SampleBatch, ModelOutput], Any]
        ] = None,
    ):
        result = []
        for i, data in enumerate(dataloader):
            data = cast(SampleBatch, data).cuda()

            with torch.no_grad():
                outputs = model.forward(data)
                if outputs.logits.shape[0] == 0:
                    print("Skipping _predict because outputs don't have logits")
                    return []

                decoded = self.decode_predictions(outputs, data)
                predicted = decoded.predictions
                targets = decoded.targets

                if handle_prediction_batch is not None:
                    handle_prediction_batch(i, data, outputs)
                combined = zip(predicted, targets)
                batch_predictions = []
                batch_result = {
                    "metrics": outputs.metrics,
                    "batch_id": i,
                    "predictions": batch_predictions,
                }

                for prediction, target in combined:
                    batch_predictions.append(
                        {
                            "prediction": prediction,
                            "target    ": target,
                        }
                    )
                result.append(batch_result)
            print(
                f"Running predictions on test. Batch {i + 1}/{len(dataloader)}\r",
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
        return optim_cls(
            self.model.parameters(),
            lr=self.base_config.learning_rate,
            weight_decay=self.base_config.weight_decay,
        )

    @abstractmethod
    def get_vocab(self) -> list[str]:
        raise NotImplementedError("Implement get_vocab in subclass")

    def visualize_predictions(
        self,
        batch: SampleBatch,
        output: ModelOutput,
        out_path: str,
        batch_id: int,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        # Assuming `predictions` is your model's output with shape (seq_len, vocab_size)
        # And `vocab` is your vocabulary list with the characters

        predictions = output.logits.softmax(-1).cpu().numpy()
        predicted_str, target_str = self.decode_predictions(output, batch)

        batch_size, seq_len, vocab_size = predictions.shape
        vocab = self.get_vocab()

        px = 1 / plt.rcParams["figure.dpi"]

        nrows = min(batch_size, 4)
        fig, _axs = plt.subplots(
            nrows=nrows,
            figsize=(
                seq_len * 18 * px,
                ((vocab_size + 1) * 1.5) * nrows * 18 * px,
            ),
        )  # Adjust figsize as needed
        norm = Normalize(
            vmin=0, vmax=1
        )  # Normalize the color scale to the probability values
        axs = _axs if batch_size > 1 else [_axs]
        for sample_index, ax in enumerate(axs):
            print("Visualizing sample", sample_index + 1, "/", len(axs))
            # Create a table
            table_data = []
            for row in range(vocab_size):
                table_row = []
                for col in range(seq_len):
                    table_row.append(f"{vocab[row]}")
                table_data.append(table_row)

            # Create the table
            table = ax.table(
                cellText=table_data,
                cellLoc="center",
                loc="center",
                cellColours=plt.cm.Blues(norm(predictions[sample_index].T)),  # type: ignore
            )

            # Highlighting cells with the highest probability in each column
            for i in range(seq_len):
                col_vals = predictions[sample_index][i, :]
                max_val_index = np.argmax(
                    col_vals
                )  # Find the index of the max probability in the column
                # Set properties for the cell with the highest probability
                table[max_val_index.item(), i].set_edgecolor(
                    "red"
                )  # Adjust for 1-based indexing in table
                table[max_val_index.item(), i].set_linewidth(2)  # Make the border bold

            # Update table properties
            table.auto_set_font_size(False)
            table.set_fontsize(8)  # Adjust font size as needed
            # ax.axis("off")  # Hide the axis
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(
                f"Target: {target_str[sample_index]}\nPrediction: {predicted_str[sample_index]}"
            )

        plt.title(f"Displaying {nrows}/{batch_size} samples")
        plt.tight_layout()
        plt.savefig(out_path)
