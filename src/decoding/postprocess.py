import re
import time
import pickle
import numpy as np
from torch.utils.data import Dataset

from edit_distance import SequenceMatcher
import torch

import matplotlib.pyplot as plt


from neural_decoder.neural_decoder_trainer import (
    getDatasetLoaders,
)
from neural_decoder.neural_decoder_trainer import loadModel
import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils
import pickle
import os
import argparse

from datasets.brain2text_w_phonemes import PhonemeSampleBatch
from model.b2tmodel import ModelOutput
from src.args.yaml_config import YamlConfig
from typing import NamedTuple, cast


class LLMOutput(NamedTuple):
    cer: list[float]
    wer: list[float]
    decoded_transcripts: list[str]
    confidences: list[float]


# Argparse postprocessing data directory
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Data directory for post processing")
args = parser.parse_args()

batches: list[tuple[tuple[PhonemeSampleBatch, ModelOutput], str]] = []
for file in os.listdir(args.data_dir):
    path = os.path.join(args.data_dir, file)
    with open(path, "rb") as handle:
        batch = (
            cast(tuple[PhonemeSampleBatch, ModelOutput], pickle.load(handle)),
            str(file),
        )
        batches.append(batch)

yaml_config = YamlConfig()

MODEL_CACHE_DIR = yaml_config.config.cache_dir
# Load OPT 6B model
llm, llm_tokenizer = lmDecoderUtils.build_opt(
    cacheDir=MODEL_CACHE_DIR, device="auto", load_in_8bit=True
)

lmDir = yaml_config.config.ngram_lm_model_path
ngramDecoder = lmDecoderUtils.build_lm_decoder(
    lmDir, acoustic_scale=0.5, nbest=100, beam=18
)


# LM decoding hyperparameters
acoustic_scale = 0.5
blank_penalty = np.log(7)
llm_weight = 0.5

llm_outputs = []
# Generate nbest outputs from 5gram LM
start_t = time.time()

for (input, output), file in batches:
    batch_logits = output.logits

    rnn_outputs = {
        "logits": [],
        "logitLengths": [],
        "transcriptions": [],
    }

    nbest_outputs = []
    for i in range(batch_logits.shape[0]):
        logits = batch_logits[i].detach().cpu().numpy()
        logits = np.concatenate(
            [logits[:, 1:], logits[:, 0:1]], axis=-1
        )  # Blank moved to the end
        logits = lmDecoderUtils.rearrange_speech_logits(
            logits[None, :, :], has_sil=True
        )
        nbest = lmDecoderUtils.lm_decode(
            ngramDecoder,
            logits[0],
            blankPenalty=blank_penalty,
            returnNBest=True,
            rescore=True,
        )
        nbest_outputs.append(nbest)

        new_trans = [ord(c) for c in input.transcriptions[i]] + [0]
        rnn_outputs["transcriptions"].append(np.array(new_trans))

        # Rescore nbest outputs with LLM
        start_t = time.time()
        print("LLM rescoring")
        llm_out = lmDecoderUtils.cer_with_gpt2_decoder(
            llm,
            llm_tokenizer,
            nbest_outputs[:],
            acoustic_scale,
            rnn_outputs,
            outputType="speech_sil",
            returnCI=True,
            lengthPenalty=0,
            alpha=llm_weight,
        )
        time_per_sample = (time.time() - start_t) / len(logits)
        print(f"LLM decoding took {time_per_sample} seconds per sample")

        llm_output = LLMOutput(
            cer=llm_out["cer"],
            wer=llm_out["wer"],
            decoded_transcripts=llm_out["decoded_transcripts"],
            confidences=llm_out["confidences"],
        )

        with open(os.path.join(args.data_dir, "decoded_" + file), "wb") as handle:
            pickle.dump(llm_output, handle)

time_per_sample = (time.time() - start_t) / len(batches)
print(f"5gram decoding took {time_per_sample} seconds per batch")
