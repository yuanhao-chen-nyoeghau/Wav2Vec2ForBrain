from collections import defaultdict, Counter
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
from torch import nn
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from typing import Literal
import transformers


def calc_seq_len(index_seq: torch.Tensor):
    for i in range(len(index_seq)):
        j = len(index_seq) - 1 - i
        if index_seq[j].item() > 0:
            return j + 1
    return 0


def prefix_beam_search(ctc, lm=None, k=25, alpha=0.30, beta=5, prune=0.001):
    """
    Performs prefix beam search on the output of a CTC network.

    Args:
            ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)
            lm (func): Language model function. Should take as input a string and output a probability.
            k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
            alpha (float): The language model weight. Should usually be between 0 and 1.
            beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
            prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.

    Retruns:
            string: The decoded CTC output.
    """

    lm = (
        (lambda l: 1) if lm is None else lm
    )  # if no LM is provided, just set to function returning 1
    W = lambda l: re.findall(r"\w+[\s|>]", l)
    alphabet = list(ascii_lowercase) + [" ", ">", "%"]
    F = ctc.shape[1]
    ctc = np.vstack(
        (np.zeros(F), ctc)
    )  # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]

    # STEP 1: Initiliazation
    O = ""
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):
        pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > prune)[0]]
        for l in A_prev:

            if len(l) > 0 and l[-1] == ">":
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue

            for c in pruned_alphabet:
                c_ix = alphabet.index(c)
                # END: STEP 2

                # STEP 3: “Extending” with a blank
                if c == "%":
                    Pb[t][l] += ctc[t][-1] * (Pb[t - 1][l] + Pnb[t - 1][l])
                # END: STEP 3

                # STEP 4: Extending with the end character
                else:
                    l_plus = l + c
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                    # END: STEP 4

                    # STEP 5: Extending with any other non-blank character and LM constraints
                    elif len(l.replace(" ", "")) > 0 and c in (" ", ">"):
                        lm_prob = lm(l_plus.strip(" >")) ** alpha
                        Pnb[t][l_plus] += (
                            lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                        )
                    else:
                        Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    # END: STEP 5

                    # STEP 6: Make use of discarded prefixes
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += ctc[t][-1] * (
                            Pb[t - 1][l_plus] + Pnb[t - 1][l_plus]
                        )
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]
                    # END: STEP 6

        # STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
        # END: STEP 7

    return A_prev[0].strip(">")


@dataclass
class GRUTrafoModelOutput:
    logits: torch.Tensor
    intermediate: torch.Tensor
    metrics: dict[str, float]
    loss: Optional[torch.Tensor] = None


class B2TGruTrafoArgsModel(B2TGruArgsModel):
    lm_checkpoint: str = "google-bert/bert-base-uncased"
    training_task: Literal["gru", "trafo", "classifier_head"] = "gru"
    language_model_mode: Literal["beam", "lm"] = "lm"


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
        if self.config.language_model_mode == "lm":
            self.lm: transformers.BertForMaskedLM = (
                transformers.AutoModelForMaskedLM.from_pretrained(
                    self.config.lm_checkpoint
                )
            )
            self.classifier_head = torch.nn.Linear(
                in_features=768, out_features=self.tokenizer.vocab_size
            )

    def language_model_forward(
        self, rnn_probs: torch.Tensor, targets: torch.Tensor | None
    ) -> GRUTrafoModelOutput:
        predicted_ids = rnn_probs.argmax(dim=-1).cpu().numpy()
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
            intermediate=rnn_probs,
            metrics={"trafo_loss": output.loss.item()} if targets != None else {},
            loss=output.loss if targets != None else None,
        )

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> GRUTrafoModelOutput:
        # GRU inference
        rnn_output = self.rnn.forward(x, targets)

        if self.config.language_model_mode == "lm":
            lm_output = self.language_model_forward(
                rnn_probs=rnn_output.logits, targets=targets
            )
            loss = (
                rnn_output.loss
                if self.config.training_task == "gru"
                else lm_output.loss
            )
            rnn_output.metrics.update(lm_output.metrics)

            return GRUTrafoModelOutput(
                logits=lm_output.logits,
                intermediate=rnn_output.logits,
                metrics=rnn_output.metrics,
                loss=loss,
            )
        else:
            np_logits = rnn_output.logits.cpu().detach().numpy()
            output_logits = []
            for i in range(np_logits.shape[0]):
                logits = np_logits[i, :, :]
                beam_strings = prefix_beam_search(logits)
                output_logits.append(
                    self.tokenizer.encode_plus(
                        beam_strings, return_tensors="pt", padding=True
                    ).input_ids
                )
            output_logits = torch.stack(output_logits, 0)
            return GRUTrafoModelOutput(
                logits=output_logits,
                intermediate=rnn_output.logits,
                metrics=rnn_output.metrics,
                loss=rnn_output.loss,
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
        self.tokenizer = self._create_tokenizer()
        updated_vocab = {
            key: (
                self.tokenizer.get_vocab()[key]
                if key in self.tokenizer.get_vocab().keys()
                else val
            )
            for (key, val) in self.wav2vec_tokenizer.get_vocab().items()
        }
        self.wav2vec_tokenizer.vocab = updated_vocab
        super().__init__(config, yamlConfig)
        self.model: B2TModel = self.model

        assert (
            self.config.preprocessing == "seperate_zscoring"
        ), "Only seperate_zscoring is currently supported"
        assert (
            self.config.training_task == "gru"
            or self.config.language_model_mode == "lm"
        ), "Can only run training task trafo without beam search"

    def get_name(self) -> str:
        return "b2t_gru+trafo_experiment"

    def get_trainable_params(self):
        if self.config.training_task == "gru":
            return self.model.rnn.parameters()
        elif self.config.training_task == "trafo":
            return [
                {"params": self.model.trafo.parameters()},
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
