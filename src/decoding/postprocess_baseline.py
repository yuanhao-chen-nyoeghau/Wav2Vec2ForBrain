import time
import pickle
import numpy as np
import pickle
import os
import argparse
from src.datasets.batch_types import PhonemeSampleBatch
from src.model.b2tmodel import ModelOutput
from src.args.yaml_config import YamlConfig
from src.decoding.decoding_types import LLMOutput

# Code from: https://github.com/fwillett/speechBCI/blob/main/AnalysisExamples/rnn_step3_baselineRNNInference.ipynb


def text_to_ascii(text: str):
    return [ord(char) for char in text]


def prepare_transcription_batch(transcriptions: list[str]):
    transcriptions = [
        t.replace("<s>", "").replace("</s>", "").lower() for t in transcriptions
    ]
    max_len = (
        max([len(t) for t in transcriptions]) + 1
    )  # make sure there is an end token/blank in each sequence
    paddedTranscripts = []
    for t in transcriptions:
        paddedTranscription = np.zeros([max_len]).astype(np.int32)
        paddedTranscription[0 : len(t)] = np.array(text_to_ascii(t))
        paddedTranscripts.append(paddedTranscription)
    return np.stack(paddedTranscripts, axis=0)


# before executing:
# export PYTHONPATH="/hpi/fs00/home/tobias.fiedler/brain2text"
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# conda install cudatoolkit -y


if __name__ == "__main__":
    print(
        "Environment running postprocessing script:",
        os.environ["CONDA_DEFAULT_ENV"],
        flush=True,
    )
    import neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils

    # Argparse postprocessing data directory
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, help="Data directory for post processing"
    )
    args = parser.parse_args()
    print(f"Converting data in {args.data_dir} to LLM outputs", flush=True)
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

    lmDir = yaml_config.config.ngram_lm_model_path
    print("Loading n-gram LM", flush=True)
    ngramDecoder = lmDecoderUtils.build_lm_decoder(
        lmDir, acoustic_scale=0.8, nbest=1, beam=18  # 1.2
    )
    print("n-gram loaded", flush=True)

    # LM decoding hyperparameters
    acoustic_scale = 0.5
    blank_penalty = np.log(2)
    llm_weight = 0.5

    llm_outputs = []
    # Generate nbest outputs from 5gram LM
    start_t = time.time()
    out_dir = os.path.join(args.data_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    for (input, output), file in batches:
        batch_logits = output.logits

        logit_lens = output.logit_lens
        assert (
            logit_lens != None
        ), "Logit lens is None (probably because the model outputs in the pickle are outdated)"

        np_logits = np.stack(
            [
                np.concatenate(
                    [
                        sample_logits[:, 1:],
                        sample_logits[:, 0:1],
                    ],
                    axis=-1,
                )
                for sample_logits in output.logits.cpu().numpy()
            ]  # move blank to the end
        )
        rnn_outputs = {
            "logits": np_logits,
            "logitLengths": logit_lens.cpu().numpy(),
            "transcriptions": prepare_transcription_batch(input.transcriptions),
        }

        decoder_out = lmDecoderUtils.cer_with_lm_decoder(
            ngramDecoder, rnn_outputs, outputType="speech_sil", blankPenalty=np.log(2)
        )

        llm_output = LLMOutput(
            cer=decoder_out["cer"],
            wer=decoder_out["wer"],
            decoded_transcripts=decoder_out["decoded_transcripts"],
            target_transcripts=input.transcriptions,
            confidences=None,
        )

        print("decoded", decoder_out["decoded_transcripts"], flush=True)
        print("target", decoder_out["true_transcripts"], flush=True)
        print("cer", decoder_out["cer"], flush=True)
        print("wer", decoder_out["wer"], flush=True)
        with open(os.path.join(out_dir, file), "wb") as handle:
            pickle.dump(llm_output, handle)
    time_per_batch = (time.time() - start_t) / len(batches)
    print(
        f"3gram decoding took {time_per_batch} seconds per batch and a total of {time.time() - start_t} seconds",
        flush=True,
    )
