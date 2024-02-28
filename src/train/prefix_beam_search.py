from collections import defaultdict, Counter
from string import ascii_lowercase
import re
import numpy as np
from tokenizers import AddedToken
import torch
import transformers


def translate_string(
    input_string: str,
    input_tokenizer: (
        transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
    ),
    other_tokenizer: (
        transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
    ),
):
    special_tokens_input = input_tokenizer.special_tokens_map_extended
    special_tokens_output = other_tokenizer.special_tokens_map_extended

    mapping: dict[str, str] = {
        str(special_tokens_input[token]): (
            str(special_tokens_output[token])
            if token in special_tokens_output.keys()
            else ""
        )
        for token in special_tokens_input.keys()
    }
    for key, val in mapping.items():
        input_string = input_string.replace(key, val)
    input_string = input_string.replace("|", " ")
    input_string = input_string.lower()
    return input_string


# Taken from https://github.com/corticph/prefix-beam-search
def prefix_beam_search(
    ctc: np.ndarray,
    lm,
    experiment_tokenizer: (
        transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
    ),
    lm_tokenizer: (
        transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
    ),
    k=25,
    alpha=0.70,
    beta=5,
    prune=0.001,
):
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

    W = lambda l: re.findall(r"\w+[\s|>]", l)
    F = ctc.shape[1]
    alphabet = list(
        dict(
            sorted(experiment_tokenizer.get_vocab().items(), key=lambda item: item[1])
        ).keys()
    )
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

            if len(l) > 0 and l[-1] == experiment_tokenizer.eos_token:
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue

            for c in pruned_alphabet:
                c_ix = alphabet.index(c)
                # END: STEP 2

                # STEP 3: “Extending” with a blank
                if c == experiment_tokenizer.pad_token:
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
                    elif len(
                        l.replace("|", "").replace("<s>", "").replace("</s>", "")
                    ) > 0 and c in (
                        "|",
                        experiment_tokenizer.eos_token,
                    ):
                        truncated_string = l_plus.strip(
                            experiment_tokenizer.eos_token
                        ).strip("|")
                        translated_string = translate_string(
                            truncated_string,
                            input_tokenizer=experiment_tokenizer,
                            other_tokenizer=lm_tokenizer,
                        )

                        # Calculate probability that next_word comes after the given prefix
                        inputs = lm_tokenizer(translated_string, return_tensors="pt")
                        output = lm(**inputs)
                        output_probs = torch.nn.functional.softmax(
                            output.logits, dim=-1
                        )
                        sentence_prob = 0.0
                        for i in range(output_probs.shape[1]):
                            token_id = int(inputs.input_ids[0, i].item())
                            token_prob = output_probs[0, i, token_id].item()
                            sentence_prob = sentence_prob + token_prob
                        # next_token_probs = output_probs[0, -1, :]
                        # lm_prob = next_token_probs[
                        #     int(inputs.input_ids[0, -1].item())
                        # ].item()
                        lm_prob = sentence_prob / output_probs.shape[1]
                        lm_prob = lm_prob**alpha
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
        Pb[t], Pnb[t] = normalize_counters(Pb[t], Pnb[t])
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
        # END: STEP 7
    if len(A_prev) > 0:
        return A_prev[0].strip(experiment_tokenizer.eos_token)
    else:
        return "Prefix search unsuccessful"


def normalize_counters(
    blank_prob_counter: Counter, no_blank_prob_counter: Counter
) -> tuple[Counter, Counter]:
    total = sum(blank_prob_counter.values()) + sum(no_blank_prob_counter.values())
    if total > 0:
        return Counter(
            {key: val / total for key, val in blank_prob_counter.items()}
        ), Counter({key: val / total for key, val in no_blank_prob_counter.items()})
    else:
        return blank_prob_counter, no_blank_prob_counter
