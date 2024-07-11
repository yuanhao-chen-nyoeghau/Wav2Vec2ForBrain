from abc import abstractmethod, ABCMeta
from ast import Not
from git import Optional
from src.datasets.base_dataset import BaseDataset
from src.datasets.batch_types import SampleBatch
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
from src.train.history import MetricEntry, SingleEpochHistory, TrainHistory
import sys
import numpy as np
from torcheval.metrics import WordErrorRate
from wandb.sdk.wandb_run import Run
from src.train.evaluator import Evaluator
import transformers

Optimizers: dict[str, Type[Optimizer]] = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
}


class Experiment(metaclass=ABCMeta):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.base_config = BaseExperimentArgsModel(**config)
        torch.manual_seed(self.base_config.seed)
        np.random.seed(self.base_config.seed)
        self.yaml_config = yamlConfig

        self.dataloader_train = self._create_dataloader(split="train")
        self.dataloader_val = self._create_dataloader(split="val")
        self.dataloader_test = self._create_dataloader(split="test")

        self.raw_config = config

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
                strict=True,
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
        if self.base_config.use_prefix_beam_search:
            self.beam_search_lm = transformers.AutoModelForCausalLM.from_pretrained(
                self.base_config.beam_search_language_model
            )
            self.beam_search_tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.base_config.beam_search_language_model,
                cache_dir=self.yaml_config.cache_dir,
                use_fast=self.base_config.use_fast_tokenizer,
            )

    def run(self):
        from src.train.train_loop import Trainer

        if self.base_config.use_wandb:
            wandb.login(key=self.yaml_config.wandb_api_key, relogin=True)

        trainer = Trainer(self)
        wandb.init(
            project=self.yaml_config.wandb_project_name,
            entity=self.yaml_config.wandb_entity,
            config=self.raw_config,
            name=self.base_config.experiment_name,
            dir=self.yaml_config.cache_dir,
            save_code=True,
            mode="online" if self.base_config.use_wandb else "disabled",
        )
        if wandb.run is None:
            raise Exception("wandb init failed. wandb.run is None")
        with wandb.run:
            wandb.watch(trainer.model)

            if not self.base_config.only_test:
                trained_model, history = trainer.train()
                torch.save(
                    trained_model.state_dict(),
                    os.path.join(self.results_dir, "model.pt"),
                )
                with open(os.path.join(self.results_dir, "history.json"), "w") as f:
                    json.dump(history.to_dict(), f, indent=5)

                self.plot_results(history)
                self.process_test_results(history.test_losses)
            else:
                test_results = self.run_real_world_test(self.model)
                if test_results != None:
                    wandb.log(trainer._get_wandb_metrics(test_results, "test"))
                    self.process_test_results(test_results)

            artifact = wandb.Artifact(name="results", type="experiment_results")
            artifact.add_dir(f"{self.results_dir}/")
            cast(Run, wandb.run).log_artifact(artifact)
            print(f"Done. Saved results to {self.results_dir}")

    def process_test_results(self, test_results: SingleEpochHistory):
        pass

    def plot_results(self, history: TrainHistory):
        history.plot(os.path.join(self.results_dir, "history.png"))

    def run_real_world_test(self, model: B2TModel):
        test_results = self._predict_and_store(model, "test")
        if self.base_config.predict_on_train == True:
            self._predict_and_store(model, "train")
        return test_results

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

    def _predict_and_store(self, model: B2TModel, mode: Literal["train", "test"]):

        def handle_evaluation_prediction_batch(
            batch_id: int,
            batch: SampleBatch,
            outputs: ModelOutput,
        ):
            if batch_id >= self.base_config.visualize_predictions_n_batches:
                return
            out_dir = os.path.join(self.results_dir, f"{mode}_predictions")
            os.makedirs(out_dir, exist_ok=True)
            print(
                f"\nVisualizing prediction batch {batch_id+1}/{self.base_config.visualize_predictions_n_batches} for {mode}..."
            )
            self.visualize_predictions(
                batch,
                outputs,
                os.path.join(out_dir, f"batch_{batch_id}.png"),
                batch_id,
            )

        prediction = self._predict(model, mode, handle_evaluation_prediction_batch)
        if prediction != None:
            with open(
                os.path.join(self.results_dir, f"{mode}_predictions.json"),
                "w",
            ) as f:
                json.dump(prediction.to_dict(), f, indent=5)
        return prediction

    def _predict(
        self,
        model: B2TModel,
        mode: Literal["train", "test"],
        handle_prediction_batch: Optional[
            Callable[[int, SampleBatch, ModelOutput], Any]
        ] = None,
    ):
        dataloader = self.dataloader_train if mode == "train" else self.dataloader_test
        evaluator = self.create_evaluator(mode)
        model.eval()
        for i, data in enumerate(dataloader):
            data = cast(SampleBatch, data).cuda()

            with torch.no_grad():
                outputs = model.forward(data)
                if outputs.logits.shape[0] == 0:
                    print("Skipping _predict because outputs don't have logits")
                    return None
                evaluator.track_batch(outputs, data)
                if handle_prediction_batch is not None:
                    handle_prediction_batch(i, data, outputs)
            print(
                f"Running predictions on {mode}. Batch {i + 1}/{len(dataloader)}\r",
                end="",
            )

        result = evaluator.evaluate()
        evaluator.clean_up()
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
            eps=self.base_config.optimizer_epsilon,
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
        vocab = self.get_vocab()
        predictions = output.logits.softmax(-1).cpu().numpy()
        predicted_str = [
            "".join([vocab[i] for i in np.argmax(p, axis=-1)]) for p in predictions
        ]
        target_str = (
            ["".join([vocab[i] for i in p]) for p in batch.target]
            if batch.target is not None
            else None
        )

        batch_size, seq_len, vocab_size = predictions.shape
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
                f"Target: {target_str[sample_index] if target_str is not None else None}\nPrediction: {predicted_str[sample_index]}"
            )

        plt.title(f"Displaying {nrows}/{batch_size} samples")
        plt.tight_layout()
        plt.savefig(out_path)

    @abstractmethod
    def create_evaluator(self, mode: Literal["train", "val", "test"]) -> Evaluator:
        raise NotImplementedError("Implement create_evaluator in subclass")
