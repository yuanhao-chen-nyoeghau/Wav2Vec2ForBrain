from src.args.yaml_config import YamlConfig
from src.datasets.brain2text_w_phonemes import Brain2TextWPhonemesDataset
import os
from pydub import AudioSegment
from elevenlabs import save
from elevenlabs.client import ElevenLabs
from typing import NamedTuple, cast
from src.model.w2v_no_encoder import Wav2Vec2WithoutTransformerModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Config,
)
import soundfile
import torch
from src.datasets.batch_types import B2tSampleBatch
from src.datasets.discriminator_dataset import (
    B2P2TBrainFeatureExtractorArgsModel,
    DiscriminatorDataset,
)


yaml_config = YamlConfig().config
working_dir = os.path.join(yaml_config.cache_dir, "latent_analysis")
wav_out = os.path.join(working_dir, "wav")

os.makedirs(working_dir, exist_ok=True)
assert (
    yaml_config.elevenlabs_api_key is not None
), "Please provide an ElevenLabs API key in the YAML config file"
client = ElevenLabs(
    api_key=yaml_config.elevenlabs_api_key,
)


def generate_speech(transcription: str, out_mp3_path: str):
    audio = client.text_to_speech.convert(
        voice_id="pMsXgVXv3BLzUgSXRplE",
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=transcription,
    )
    save(audio, out_mp3_path)


def convert_mp3_to_wav(mp3_path: str, wav_path: str):
    audio = AudioSegment.from_file(mp3_path, format="mp3", frame_rate=44100)
    audio = audio.set_frame_rate(16000)
    audio.export(wav_path, format="wav")


def generate_speech_multiple(transcriptions: list[str]):
    mp3_out = os.path.join(working_dir, "mp3")
    os.makedirs(mp3_out, exist_ok=True)
    os.makedirs(wav_out, exist_ok=True)

    def generate_wav(index: int, override_existing: bool = False) -> bool:
        transcription = transcriptions[index]
        mp3_path = os.path.join(mp3_out, f"{index}.mp3")
        newly_created = False
        if not os.path.exists(mp3_path) or override_existing:
            generate_speech(transcription, mp3_path)
            newly_created = True
        wav_path = os.path.join(wav_out, f"{index}.wav")
        if not os.path.exists(wav_path) or override_existing:
            convert_mp3_to_wav(mp3_path, wav_path)
        return newly_created

    num_generated = 0
    for i in range(len(transcriptions)):
        newly_generated = generate_wav(i)
        if newly_generated:
            num_generated += 1
        print(
            f"\r{i+1}/{len(transcriptions)} audios generated ({num_generated} newly generated)",
            end="",
        )
    print("\n")


class LatentRepresentation(NamedTuple):
    idx: int  # Index of the corresponding transcription in the b2t test dataset
    latent: torch.Tensor


class Representations(NamedTuple):
    non_aggregated: list[LatentRepresentation]
    aggregated: list[LatentRepresentation]


def generate_audio_representations(ds: Brain2TextWPhonemesDataset) -> Representations:
    def load_wav_as_tensor(path: str):
        wav = soundfile.read(path)
        return torch.tensor(wav[0], dtype=torch.float32)

    w2v_config = cast(
        Wav2Vec2Config,
        Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h"),
    )

    w2v_feature_extractor = cast(
        Wav2Vec2WithoutTransformerModel,
        Wav2Vec2WithoutTransformerModel.from_pretrained(
            "facebook/wav2vec2-base-960h",
            config=w2v_config,
            cache_dir=yaml_config.cache_dir,
        ),
    )
    w2v_feature_extractor.eval()
    non_aggregated = []
    aggregated = []
    with torch.no_grad():
        num_generated = 0
        for i, filename in enumerate(os.listdir(wav_out)):
            idx = int(filename.split(".")[0])
            if idx >= len(ds):
                continue
            wav_path = os.path.join(wav_out, filename)
            wav_tensor = load_wav_as_tensor(wav_path)
            features = w2v_feature_extractor.forward(wav_tensor.unsqueeze(0)).squeeze()

            for timestamp in features:
                non_aggregated.append(LatentRepresentation(idx, timestamp))
            aggregated.append(
                LatentRepresentation(idx, torch.mean(features.squeeze(), dim=0))
            )
            num_generated += 1
            print(
                f"\r{num_generated}/{len(ds)} audio representations generated ", end=""
            )
    print("\n")
    assert None not in non_aggregated
    return Representations(non_aggregated, aggregated)


def generate_brain_representations(ds: Brain2TextWPhonemesDataset) -> Representations:
    config = B2P2TBrainFeatureExtractorArgsModel(
        encoder_fc_hidden_sizes=[512],
        encoder_gru_hidden_size=1024,
        encoder_num_gru_layers=1,
    )

    brain_encoder = DiscriminatorDataset.brain_feature_extractor_from_config(
        config,
        "/hpi/fs00/scratch/tobias.fiedler/brain2text/experiment_results/b2p2t_gru+w2v/2024-08-20_11#57#39/brain_encoder.pt",
        "facebook/wav2vec2-base-960h",
    )
    brain_encoder.eval()
    non_aggregated = []
    aggregated = []
    with torch.no_grad():
        for idx in range(len(ds)):
            batch = B2tSampleBatch(ds[idx].input.unsqueeze(0), None)
            batch.day_idxs = torch.tensor(ds[idx].day_idx).unsqueeze(0)
            features = (
                brain_encoder.forward(batch.cuda()).logits.squeeze().detach().cpu()
            )
            for timestamp in features:
                non_aggregated.append(LatentRepresentation(idx, timestamp))
            aggregated.append(LatentRepresentation(idx, torch.mean(features, dim=0)))
            print(f"\r{idx+1}/{len(ds)} brain representations generated ", end="")

    print("\n")
    assert None not in non_aggregated
    return Representations(non_aggregated, aggregated)
