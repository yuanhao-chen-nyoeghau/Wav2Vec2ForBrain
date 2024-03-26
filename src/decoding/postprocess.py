import time
import pickle
import numpy as np
import pickle
import os
import argparse

from src.decoding.decoding_types import LLMOutput, prepare_transcription_batch
from src.datasets.batch_types import PhonemeSampleBatch
from src.model.b2tmodel import ModelOutput
from src.args.yaml_config import YamlConfig


# before executing:
# export PYTHONPATH="/hpi/fs00/home/tobias.fiedler/brain2text"
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# conda install cudatoolkit -y


LLMOutputBatch = list[LLMOutput]


if __name__ == "__main__":
    print("Environment running postprocessing script:", os.environ["CONDA_DEFAULT_ENV"])
    import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils

    # Argparse postprocessing data directory
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, help="Data directory for post processing"
    )
    parser.add_argument("--rescoring", action="store_true")
    args = parser.parse_args()
    print(f"Converting data in {args.data_dir} to LLM outputs")
    batches: list[tuple[tuple[PhonemeSampleBatch, ModelOutput], str]] = []
    for file in os.listdir(args.data_dir):
        if os.path.isdir(os.path.join(args.data_dir, file)):
            continue
        path = os.path.join(args.data_dir, file)
        with open(path, "rb") as handle:
            batch = (
                pickle.load(handle),
                str(file),
            )
            batches.append(batch)

    yaml_config = YamlConfig()

    MODEL_CACHE_DIR = yaml_config.config.cache_dir
    # Load OPT 6B model
    print("Loading LLM model")
    llm, llm_tokenizer = lmDecoderUtils.build_opt(
        cacheDir=MODEL_CACHE_DIR, device="auto", load_in_8bit=True
    )

    lmDir = yaml_config.config.ngram_lm_model_path
    print("Loading n-gram LM")
    ngramDecoder = lmDecoderUtils.build_lm_decoder(
        lmDir, acoustic_scale=0.8, nbest=10, beam=18
    )
    print("n-gram loaded")

    # LM decoding hyperparameters
    acoustic_scale = 0.5
    blank_penalty = np.log(7)
    llm_weight = 0.5

    llm_outputs = []
    # Generate nbest outputs from 5gram LM
    start_t = time.time()
    out_dir = os.path.join(args.data_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    for (input, output), file in batches:
        batch_logits = output.logits
        assert output.logit_lens != None
        rnn_outputs = {
            "transcriptions": prepare_transcription_batch(input.transcriptions),
        }

        batch_nbest_outputs = []

        out_batch: LLMOutputBatch = []
        for i in range(batch_logits.shape[0]):
            logits = batch_logits[i].detach().cpu().numpy()
            logits = np.concatenate(
                [logits[:, 1:], logits[:, 0:1]], axis=-1
            )  # Blank must be last token
            logits = lmDecoderUtils.rearrange_speech_logits(
                logits[None, :, :], has_sil=True
            )

            print("Decoding sample", i, "with rescoring enabled: ", args.rescoring)

            nbest = lmDecoderUtils.lm_decode(
                ngramDecoder,
                logits[0, : output.logit_lens[i].item()],
                blankPenalty=blank_penalty,
                returnNBest=True,
                rescore=args.rescoring == True,
            )
            batch_nbest_outputs.append(nbest)

        # Rescore nbest outputs with LLM
        start_t = time.time()
        print("LLM rescoring")
        llm_out = lmDecoderUtils.cer_with_gpt2_decoder(
            llm,
            llm_tokenizer,
            batch_nbest_outputs[:],
            acoustic_scale,
            rnn_outputs,
            outputType="speech_sil",
            returnCI=True,
            lengthPenalty=0,
            alpha=llm_weight,
        )
        time_per_batch = (time.time() - start_t) / len(logits)
        print(f"LLM decoding took {time_per_batch} seconds per sample")
        llm_output = LLMOutput(
            cer=llm_out["cer"],
            wer=llm_out["wer"],
            decoded_transcripts=llm_out["decoded_transcripts"],
            confidences=llm_out["confidences"],
            target_transcripts=input.transcriptions,
        )
        out_batch.append(llm_output)
        print("decoded", llm_out["decoded_transcripts"])
        print("target", input.transcriptions[i])

        with open(os.path.join(out_dir, file), "wb") as handle:
            pickle.dump(out_batch, handle)
    time_per_batch = (time.time() - start_t) / len(batches)
    print(f"3gram decoding took {time_per_batch} seconds per batch")
