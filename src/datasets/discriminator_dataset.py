from typing import Callable, Literal, NamedTuple, Optional

from src.model.w2v_no_encoder import Wav2Vec2WithoutTransformerModel
from src.args.base_args import B2TDatasetArgsModel
from src.datasets.batch_types import PhonemeSampleBatch, SampleBatch
from src.model.b2p2t_model import (
    DEFAULT_UNFOLDER_KERNEL_LEN,
    B2P2TModel,
    B2P2TModelArgsModel,
)
from src.model.b2p_suc import BrainEncoder, BrainEncoderArgsModel
from src.model.b2tmodel import B2TModel, ModelOutput
from src.model.w2v_suc_ctc_model import (
    W2VSUC_CTCArgsModel,
    W2VSUCForCtcModel,
)
from src.datasets.audio_with_phonemes_seq import AudioWPhonemesDatasetArgsModel
from src.datasets.brain2text_w_phonemes import Brain2TextWPhonemesDataset
from src.datasets.timit_a2p_seq_dataset import (
    TimitA2PSeqDataset,
    TimitA2PSeqSampleBatch,
)
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset
import torch
from torch.utils.data import DataLoader


class BrainEncoderWrapper(B2TModel):
    def __init__(
        self, config: BrainEncoderArgsModel, wav2vec_checkpoint: str, in_size: int
    ):
        super().__init__()
        self.encoder = BrainEncoder(
            config,
            in_size,
            wav2vec_checkpoint,
        )

    def forward(self, batch: SampleBatch) -> ModelOutput:
        out = self.encoder(batch)
        return ModelOutput(logits=out, metrics={})


class DiscriminatorSample(NamedTuple):
    hidden_state: torch.Tensor
    label: torch.Tensor


# Best checkpoint run: https://wandb.ai/machine-learning-hpi/brain2text/runs/hx7z3v6n?nw=nwusertfiedlerdev
# Local: /hpi/fs00/scratch/tobias.fiedler/brain2text/experiment_results/b2p_suc/2024-07-29_07#16#02/model.pt


class B2P2TBrainFeatureExtractorArgsModel(BrainEncoderArgsModel, B2P2TModelArgsModel):
    pass


class DiscriminatorDatasetArgsModel(B2P2TBrainFeatureExtractorArgsModel):
    brain_encoder_path: str


class DiscriminatorDataset(BaseDataset):
    def __init__(
        self,
        config: DiscriminatorDatasetArgsModel,
        yaml_config: YamlConfigModel,
        split: Literal["train", "val", "test"],
        wav2vec_checkpoint: str,
        w2v_feature_extractor: Optional[Wav2Vec2WithoutTransformerModel] = None,
        brain_feat_extractor: Optional[B2P2TModel] = None,
    ):
        super().__init__()
        audio = TimitA2PSeqDataset(
            AudioWPhonemesDatasetArgsModel(), yaml_config, split=split
        )
        brain = Brain2TextWPhonemesDataset(
            B2TDatasetArgsModel(limit_samples=None), yaml_config, split=split
        )

        w2v_feat_extractor = (
            DiscriminatorDataset.w2v_feature_extractor()
            if w2v_feature_extractor is None
            else w2v_feature_extractor
        )
        brain_feat_extractor = (
            DiscriminatorDataset.brain_feature_extractor_from_config(
                config, config.brain_encoder_path, wav2vec_checkpoint
            )
            if brain_feat_extractor is None
            else brain_feat_extractor
        )

        self.samples: list[DiscriminatorSample] = []

        audio_loader: DataLoader[TimitA2PSeqSampleBatch] = DataLoader(
            audio, batch_size=32, collate_fn=audio.get_collate_fn()
        )
        brain_loader = DataLoader(
            brain, batch_size=32, collate_fn=brain.get_collate_fn()
        )
        print("[DiscriminatorDataset] Generating samples for discriminator set ", split)

        with torch.no_grad():
            print("[DiscriminatorDataset] --> W2V feature extractor samples")
            n_w2v_batches = len(audio_loader)
            for i, audio_batch in enumerate(audio_loader):
                audio_batch: TimitA2PSeqSampleBatch = audio_batch
                hidden_state_batch = w2v_feat_extractor.forward(
                    audio_batch.input.cuda()
                )
                for hidden_states in hidden_state_batch:
                    for hidden_state in hidden_states:
                        self.samples.append(
                            DiscriminatorSample(hidden_state.cpu(), torch.tensor(0.0))
                        )

                print(
                    f"[DiscriminatorDataset] [W2V Feature extractor] Batch {i + 1}/{n_w2v_batches}\r",
                    end="",
                )
            self.n_w2v_samples = len(self.samples)
            print("\n[DiscriminatorDataset] --> Brain encoder samples")
            n_brainencoder_batches = len(brain_loader)
            for i, brain_batch in enumerate(brain_loader):
                brain_batch: PhonemeSampleBatch = brain_batch
                out = brain_feat_extractor.forward(brain_batch.cuda())
                hidden_state_batch = out.logits
                for hidden_states in hidden_state_batch:
                    for hidden_state in hidden_states:
                        self.samples.append(
                            DiscriminatorSample(hidden_state.cpu(), torch.tensor(1.0))
                        )
                print(
                    f"[DiscriminatorDataset] [Brain encoder] Batch {i + 1}/{n_brainencoder_batches}\r",
                    end="",
                )
            print("\n[DiscriminatorDataset] Done")
        self.n_brain_samples = len(self.samples) - self.n_w2v_samples

    def __getitem__(self, index: int) -> DiscriminatorSample:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    def get_collate_fn(self) -> Callable[[list[DiscriminatorSample]], SampleBatch]:
        def collate(samples: list[DiscriminatorSample]) -> SampleBatch:
            hidden_states = torch.stack([sample.hidden_state for sample in samples])
            labels = torch.stack([torch.tensor([sample.label]) for sample in samples])
            return SampleBatch(hidden_states, labels)

        return collate

    @classmethod
    def brain_feature_extractor_from_config(
        cls,
        config: B2P2TBrainFeatureExtractorArgsModel,
        brain_encoder_path: Optional[str],
        wav2vec_checkpoint: str,
    ):
        brain_feat_extractor = B2P2TModel(
            config,
            BrainEncoderWrapper(
                config,
                wav2vec_checkpoint,
                B2P2TModel.get_in_size_after_preprocessing(config.unfolder_kernel_len),
            ),
        ).cuda()
        if brain_encoder_path != None:
            state = torch.load(brain_encoder_path, map_location="cuda")
            unneeded_keys = [
                key
                for key in state.keys()
                if key.startswith("neural_decoder.discriminator")
                or key.startswith("neural_decoder.suc_for_ctc")
            ]
            for key in unneeded_keys:
                del state[key]
            brain_feat_extractor.load_state_dict(
                state,
                strict=True,
            )
        return brain_feat_extractor

    @classmethod
    def w2v_feature_extractor(cls):
        return W2VSUCForCtcModel(W2VSUC_CTCArgsModel()).cuda().w2v_feature_extractor
