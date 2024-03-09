import os
from torch.optim.optimizer import Optimizer
from src.datasets.base_dataset import Sample, SampleBatch
from src.model.b2tmodel import ModelOutput
from src.args.base_args import B2TArgsModel
from src.datasets.brain2text import Brain2TextDataset
from src.experiments.experiment import DecodedPredictionBatch, Experiment
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, cast
from transformers import AutoTokenizer
import torch
from torch.nn.functional import pad
import re
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader


class B2TExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        self.tokenizer = cast(PreTrainedTokenizer, self._create_tokenizer())
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def get_args_model():
        return B2TArgsModel

    def _create_tokenizer(self):
        if self.config.tokenizer == "wav2vec_pretrained":
            assert (
                not self.config.tokenizer_checkpoint is None
            ), "Tokenizer checkpoint (--tokenizer_checkpoint) must be set when using --tokenizer=wav2vec_pretrained"

            return AutoTokenizer.from_pretrained(
                self.config.tokenizer_checkpoint,
                cache_dir=self.yaml_config.cache_dir,
            )
        raise Exception(f"Tokenizer {self.config.tokenizer} not supported yet")

    def _create_model(self):
        raise NotImplementedError()

    def decode_predictions(
        self, predictions: ModelOutput, sample: SampleBatch
    ) -> DecodedPredictionBatch:
        predicted_ids = predictions.logits.argmax(dim=-1).cpu().numpy()
        predicted_strings = self.tokenizer.batch_decode(
            predicted_ids, group_tokens=True
        )
        label_strings = self.tokenizer.batch_decode(
            sample.target.cpu().numpy(), group_tokens=False
        )
        return DecodedPredictionBatch(predicted_strings, label_strings)

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            return self.model.parameters()

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)

    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        return Brain2TextDataset(
            config=self.config,
            yaml_config=self.yaml_config,
            split=split,
        )

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        ds = self._create_dataset(split)
        return DataLoader(
            self._create_dataset(split),
            batch_size=self.base_config.batch_size,
            shuffle=True,
            collate_fn=ds.get_collate_fn(self.tokenizer),
        )

    def visualize_predictions(
        self,
        inputs: torch.Tensor,
        output: ModelOutput,
        target_batch: list[str],
        out_path: str,
        batch_id: int,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        # Assuming `predictions` is your model's output with shape (seq_len, vocab_size)
        # And `vocab` is your vocabulary list with the characters

        predictions = output.logits.softmax(-1).cpu().numpy()
        predicted_ids = output.logits.argmax(dim=-1).cpu().numpy()
        predicted_str = self.tokenizer.batch_decode(predicted_ids)

        batch_size, seq_len, vocab_size = predictions.shape
        vocab = self.tokenizer.convert_ids_to_tokens(
            list(range(self.tokenizer.vocab_size))
        )

        px = 1 / plt.rcParams["figure.dpi"]
        fig, _axs = plt.subplots(
            nrows=min(batch_size, 32),
            figsize=(
                seq_len * 18 * px,
                ((vocab_size + 1) * 1.5) * batch_size * 18 * px,
            ),
        )  # Adjust figsize as needed
        norm = Normalize(
            vmin=0, vmax=1
        )  # Normalize the color scale to the probability values
        axs = _axs if batch_size > 1 else [_axs]
        for sample_index, ax in enumerate(axs):
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
                f"Target: {target_batch[sample_index]}\nPrediction: {predicted_str[sample_index]}"
            )

        plt.title(f"Displaying {len(axs)}/{batch_size} samples")
        plt.tight_layout()
        plt.savefig(out_path)

    def handle_evaluation_prediction_batch(
        self,
        batch_id: int,
        inputs: torch.Tensor,
        outputs: ModelOutput,
        targets: list[str],
        out_file_prefix: str,
    ):
        if batch_id >= self.base_config.visualize_predictions_n_batches:
            return
        out_dir = os.path.join(self.results_dir, f"{out_file_prefix}_predictions")
        os.makedirs(out_dir, exist_ok=True)
        print(
            f"\nVisualizing prediction {batch_id+1}/{self.base_config.visualize_predictions_n_batches} for {out_file_prefix}..."
        )
        self.visualize_predictions(
            inputs,
            outputs,
            targets,
            os.path.join(out_dir, f"batch_{batch_id}.png"),
            batch_id,
        )
