from collections import defaultdict, Counter
from fnmatch import translate
from string import ascii_lowercase
import re
import numpy as np
import re

import tokenizers
from src.experiments.b2t_gru_experiment import B2TGruArgsModel, GRUModel
from src.experiments.b2t_experiment import B2TExperiment
from torch.nn.functional import pad
from dataclasses import dataclass
from src.model.b2tmodel import B2TModel, ModelOutput
from src.args.yaml_config import YamlConfig, YamlConfigModel
from typing import Optional
import torch
from torch import LongTensor, nn
from transformers import (
    GenerationConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    HammingDiversityLogitsProcessor,
    BeamSearchScorer,
)
from torch.utils.data import DataLoader
from typing import Literal
import transformers


def calc_seq_len(index_seq: torch.Tensor):
    for i in range(len(index_seq)):
        j = len(index_seq) - 1 - i
        if index_seq[j].item() > 0:
            return j + 1
    return 0


@dataclass
class GRUTrafoModelOutput:
    logits: torch.Tensor
    intermediate: torch.Tensor
    metrics: dict[str, float]
    loss: Optional[torch.Tensor] = None


class B2TGruTrafoArgsModel(B2TGruArgsModel):
    lm_checkpoint: str = "t5-base"  # "google-bert/bert-base-uncased"
    training_task: Literal["gru", "trafo", "classifier_head"] = "gru"


class GRUTrafoModel(B2TModel):
    def __init__(
        self,
        config: B2TGruTrafoArgsModel,
        wav2vec_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        lm_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = lm_tokenizer
        self.wav2vec_tokenizer = wav2vec_tokenizer

        self.rnn = GRUModel(config=config, tokenizer=self.wav2vec_tokenizer)
        self.lm = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.config.lm_checkpoint
        )
        self.classifier_head = torch.nn.Linear(
            in_features=768, out_features=self.tokenizer.vocab_size
        )

    def translateFromWav2VecToLM(self, wav2vec_str: str) -> str:
        translations = {
            "<s>": "",
            "</s>": "",
            "<pad>": "",
            "<unk>": "",
        }
        translated = wav2vec_str.lower()
        for key, val in translations.items():
            translated = translated.replace(key, val)
        translated = translated.strip()
        return translated

    def calculatePathProbability(self, probs: torch.Tensor, word: str) -> float:
        prob = 1.0
        i = 0
        for token in self.tokenizer.encode(word):
            prob = prob * probs[0, i, token].item()
        return prob

    def language_model_forward(
        self, rnn_logits: torch.Tensor, targets: torch.Tensor | None
    ) -> GRUTrafoModelOutput:
        predicted_ids = rnn_logits.argmax(dim=-1).cpu().numpy()
        predicted = self.wav2vec_tokenizer.batch_decode(predicted_ids)

        def extract_string(input: str):
            pattern = r"<s>(.*?)</s>"
            match = re.search(pattern, input)
            if match:
                return match.group(1)
            else:
                return input.replace("<s>", "")

        predicted = [extract_string(predicted_str) for predicted_str in predicted]

        # Language model
        encoding_lm = self.tokenizer.batch_encode_plus(
            predicted, return_tensors="pt", padding=True
        )
        encoded_predictions = encoding_lm.input_ids.cuda()
        encoding_attention_mask = encoding_lm.attention_mask.cuda()
        if targets is not None:
            target_strings = self.wav2vec_tokenizer.batch_decode(
                targets.cpu().numpy(), group_tokens=False
            )
            target_strings = [
                extract_string(target_string) for target_string in target_strings
            ]
            encoded_targets = self.tokenizer.batch_encode_plus(
                target_strings, return_tensors="pt", padding=True
            ).input_ids.cuda()

            if encoded_predictions.size(1) > encoded_targets.size(1):
                encoded_predictions = encoded_predictions[:, : encoded_targets.size(1)]
                encoding_attention_mask = encoding_attention_mask[
                    :, : encoded_targets.size(1)
                ]

            elif encoded_predictions.size(1) < encoded_targets.size(1):
                padding_size = encoded_targets.size(1) - encoded_predictions.size(1)
                padding = torch.zeros(
                    encoded_predictions.size(0), padding_size, dtype=torch.int64
                ).cuda()
                encoded_predictions = torch.cat([encoded_predictions, padding], dim=1)
                encoding_attention_mask = torch.cat(
                    [encoding_attention_mask, padding], dim=1
                )

            output = self.lm(
                input_ids=encoded_predictions,
                attention_mask=encoding_attention_mask,
                labels=encoded_targets,
            )
        else:
            output = self.lm(
                input_ids=encoded_predictions,
                attention_mask=encoding_attention_mask,
            )

        return GRUTrafoModelOutput(
            logits=output.logits,
            intermediate=rnn_logits,
            metrics={"lm_loss": output.loss.item()} if targets != None else {},
            loss=output.loss if targets != None else None,
        )

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> GRUTrafoModelOutput:
        # GRU inference
        rnn_output = self.rnn.forward(x, targets)
        lm_output = self.language_model_forward(
            rnn_logits=rnn_output.logits, targets=targets
        )
        loss = rnn_output.loss if self.config.training_task == "gru" else lm_output.loss
        rnn_output.metrics.update(lm_output.metrics)

        return GRUTrafoModelOutput(
            logits=lm_output.logits,
            intermediate=rnn_output.logits,
            metrics=rnn_output.metrics,
            loss=loss,
        )


class B2tGruTrafoExperiment(B2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        self.yaml_config = YamlConfig().config
        self.wav2vec_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.tokenizer_checkpoint,
            cache_dir=self.yaml_config.cache_dir,
            use_fast=self.config.use_fast_tokenizer,
        )
        super().__init__(config, yamlConfig)
        self.model: GRUTrafoModel = self.model

        assert (
            self.config.preprocessing == "seperate_zscoring"
        ), "Only seperate_zscoring is currently supported"

    def get_name(self) -> str:
        return "b2t_gru+trafo_experiment"

    def get_trainable_params(self):
        if self.config.training_task == "gru":
            return self.model.rnn.parameters()
        elif self.config.training_task == "trafo":
            return [
                {"params": self.model.lm.parameters()},
                {"params": self.model.classifier_head.parameters()},
            ]
        else:
            return self.model.classifier_head.parameters()

    @staticmethod
    def get_args_model():
        return B2TGruTrafoArgsModel

    def _create_model(self):
        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        model = GRUTrafoModel(self.config, self.wav2vec_tokenizer, self.tokenizer)
        return model

    def _create_tokenizer(self):
        return transformers.AutoTokenizer.from_pretrained(self.config.lm_checkpoint)

    def get_collate_fn(self):
        multiple_channels = self.config.preprocessing == "seperate_zscoring_2channels"

        def _collate(batch: list[tuple[torch.Tensor, str]]):
            max_block_len = max(
                [x.size(1 if multiple_channels else 0) for x, _ in batch]
            )
            padded_blocks = [
                pad(
                    x,
                    (0, 0, 0, max_block_len - x.size(1 if multiple_channels else 0)),
                    mode="constant",
                    value=0,
                )
                for x, _ in batch
            ]

            def process_label(label: str) -> str:
                if self.config.remove_punctuation:
                    chars_to_ignore_regex = r'[\,\?\.\!\-\;\:"]'
                    label = re.sub(chars_to_ignore_regex, "", label)
                # label = label.upper()
                return label

            # TODO: enable putting in different tokenizer
            batch_label_ids: list[list[int]] = self.wav2vec_tokenizer(
                [process_label(label) for _, label in batch],
                padding="longest",
                return_tensors="pt",
            ).input_ids

            return torch.stack(padded_blocks), batch_label_ids

        return _collate

    def _predict(self, model: GRUTrafoModel, dataloader: DataLoader):
        result = []
        for i, data in enumerate(dataloader):
            inputs, labels = data

            with torch.no_grad():
                outputs = model.forward(inputs.cuda(), labels.cuda())
                if outputs.logits.shape[0] == 0:
                    print("Skipping _predict because outputs don't have logits")
                    return []
                predicted_ids = outputs.logits.argmax(dim=-1).cpu().numpy()
                predicted = self.tokenizer.batch_decode(predicted_ids)

                intermediate_ids = outputs.intermediate.argmax(dim=-1).cpu().numpy()
                intermediate = self.wav2vec_tokenizer.batch_decode(intermediate_ids)

                result.append(
                    {
                        "predicted": predicted,
                        "intermediate": intermediate,
                        "label": self.wav2vec_tokenizer.batch_decode(
                            labels.cpu().numpy(), group_tokens=False
                        ),
                    }
                )
            print(
                f"Running predictions on test. Batch {i + 1}/{len(dataloader)}\r",
                end="",
            )
        return result

    def batch_decode(self, batch: torch.Tensor):
        return self.wav2vec_tokenizer.batch_decode(
            batch.cpu().numpy(), group_tokens=False
        )
