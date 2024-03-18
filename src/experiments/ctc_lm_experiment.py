from src.datasets.batch_types import SampleBatch
from src.datasets.ctc_text_dataset import CTCTextDataset
from src.experiments.experiment import Experiment
from src.args.wav2vec_args import ACTIVATION_FUNCTION
from src.args.base_args import (
    BaseExperimentArgsModel,
    CTCTextDatasetArgsModel,
)
from src.model.b2tmodel import B2TModel
from src.args.yaml_config import YamlConfigModel
from typing import Any, Callable, Literal, Optional, cast
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.functional import pad
import re
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from src.model.b2tmodel import B2TModel, ModelOutput


class CtcLmArgsModel(BaseExperimentArgsModel, CTCTextDatasetArgsModel):
    ctclm_nhead: int = 8
    ctclm_num_layers: int = 6
    ctclm_classifier_hidden_sizes: list[int] = []
    ctclm_classifier_activation: ACTIVATION_FUNCTION = "gelu"
    ctclm_d_model: int = 128
    tokenizer: Literal["wav2vec_pretrained", "ours"] = "wav2vec_pretrained"
    tokenizer_checkpoint: Literal["facebook/wav2vec2-base-100h", None] = (
        "facebook/wav2vec2-base-100h"
    )


class CtcLmExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        self.yaml_config = yamlConfig
        self.tokenizer = self._create_tokenizer()
        self.dataset = CTCTextDataset(self.config, yamlConfig, self.tokenizer)

        super().__init__(config, yamlConfig)
        self.model: B2TModel = self.model

        self.config: CtcLmArgsModel = self.config

    def get_name(self) -> str:
        return "ctc_lm_experiment"

    @staticmethod
    def get_args_model():
        return CtcLmArgsModel

    def _create_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        from src.model.ctc_lm_model import CTCLMModel

        model = CTCLMModel(self.config, self.tokenizer)
        return model

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> Dataset:
        return self.dataset.get_split(split)

    def _create_tokenizer(self) -> PreTrainedTokenizer:
        if self.config.tokenizer == "wav2vec_pretrained":
            assert (
                not self.config.tokenizer_checkpoint is None
            ), "Tokenizer checkpoint (--tokenizer_checkpoint) must be set when using --tokenizer=wav2vec_pretrained"

            return cast(
                PreTrainedTokenizer,
                AutoTokenizer.from_pretrained(
                    self.config.tokenizer_checkpoint,
                    cache_dir=self.yaml_config.cache_dir,
                ),
            )
        raise Exception(f"Tokenizer {self.config.tokenizer} not supported yet")

    def _predict(
        self,
        model: B2TModel,
        dataloader: DataLoader,
        handle_prediction_batch: Optional[
            Callable[[int, torch.Tensor, ModelOutput, list[str]], Any]
        ] = None,
    ):
        result = []
        for i, data in enumerate(dataloader):
            data = cast(SampleBatch, data).cuda()
            inputs, labels = data
            with torch.no_grad():
                outputs = model.forward(data)
                if outputs.logits.shape[0] == 0:
                    print("Skipping _predict because outputs don't have logits")
                    return []
                predicted_ids = outputs.logits.argmax(dim=-1).cpu().numpy()
                inputs_decoded = self.tokenizer.batch_decode(
                    inputs.argmax(dim=-1).cpu().numpy()
                )
                predicted = self.tokenizer.batch_decode(predicted_ids)
                targets = self.tokenizer.batch_decode(
                    labels.cpu().numpy(), group_tokens=False
                )
                if handle_prediction_batch is not None:
                    handle_prediction_batch(i, inputs, outputs, targets)
                combined = zip(predicted, targets, inputs_decoded)
                batch_predictions = []
                batch_result = {
                    "metrics": outputs.metrics,
                    "batch_id": i,
                    "predictions": batch_predictions,
                }

                for prediction, target, input in combined:
                    batch_predictions.append(
                        {
                            "prediction": prediction,
                            "target    ": target,
                            "input": input,
                        }
                    )
                result.append(batch_result)
            print(
                f"Running predictions on test. Batch {i + 1}/{len(dataloader)}\r",
                end="",
            )
        return result

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
        import os

        # Assuming `predictions` is your model's output with shape (seq_len, vocab_size)
        # And `vocab` is your vocabulary list with the characters

        predictions = output.logits.softmax(-1).cpu().numpy()
        predicted_ids = output.logits.argmax(dim=-1).cpu().numpy()
        predicted_str = self.tokenizer.batch_decode(predicted_ids)
        input_str = self.tokenizer.batch_decode(batch.input.argmax(-1).cpu().numpy())
        target_str = self.tokenizer.batch_decode(
            batch.target.cpu().numpy(), group_tokens=False
        )

        batch_size, seq_len, vocab_size = predictions.shape
        vocab = self.tokenizer.convert_ids_to_tokens(
            list(range(self.tokenizer.vocab_size))
        )

        px = 1 / plt.rcParams["figure.dpi"]

        batch_out_dir = os.path.join(os.path.dirname(out_path), f"batch_{batch_id}")
        os.makedirs(batch_out_dir, exist_ok=True)

        for sample_index in range(batch_size):
            print(
                f"Visualizing sample {sample_index+1}/{batch_size} of batch {batch_id+1}/{self.config.visualize_predictions_n_batches}\r",
                end="",
            )
            sample_out_path = os.path.join(batch_out_dir, f"{sample_index}.png")

            fig, axs = plt.subplots(
                nrows=2,
                figsize=(
                    seq_len * 18 * px,
                    ((vocab_size + 1)) * 1.4 * 2 * 18 * px,
                ),
            )  # Adjust figsize as needed
            norm = Normalize(
                vmin=0, vmax=1
            )  # Normalize the color scale to the probability values
            for i, ax in enumerate(axs):
                data = (
                    batch.input[sample_index] if i == 0 else predictions[sample_index]
                )
                str_data = (
                    input_str[sample_index] if i == 0 else predicted_str[sample_index]
                )
                title = "Input" if i == 0 else "Prediction"
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
                    cellColours=plt.cm.Blues(norm(data.T)),  # type: ignore
                )

                # Highlighting cells with the highest probability in each column
                for i in range(seq_len):
                    col_vals = data[i, :]
                    max_val_index = np.argmax(
                        col_vals
                    )  # Find the index of the max probability in the column
                    # Set properties for the cell with the highest probability
                    table[max_val_index.item(), i].set_edgecolor(
                        "red"
                    )  # Adjust for 1-based indexing in table
                    table[max_val_index.item(), i].set_linewidth(
                        2
                    )  # Make the border bold

                # Update table properties
                table.auto_set_font_size(False)
                table.set_fontsize(8)  # Adjust font size as needed
                ax.axis("off")  # Hide the axis
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Tokenized {title}: {str_data}")

            fig.suptitle(
                f"Target: {target_str[sample_index]}",
                fontsize="large",
                fontweight="bold",
            )
            plt.savefig(sample_out_path)

    def get_vocab(self) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(
            list(range(self.tokenizer.vocab_size))
        )
