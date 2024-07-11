import math
from pydantic import BaseModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Config,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2BaseModelOutput,
)
import torch
from torch import log_softmax, nn
from typing import Literal, Optional, Tuple, Union, cast

from src.args.wav2vec_args import ACTIVATION_FUNCTION
from datasets.timit_seq_dataset import Phoneme, TimitSeqSampleBatch
from src.datasets.batch_types import PhonemeSampleBatch
from src.datasets.brain2text_w_phonemes import PHONE_DEF_SIL
from src.model.b2tmodel import B2TModel, ModelOutput
from src.util.nn_helper import create_fully_connected


class W2VSUCArgsModel(BaseModel):
    suc_hidden_sizes: list[int] = []
    suc_hidden_activation: ACTIVATION_FUNCTION = "gelu"
    suc_dropout: float = 0.0
    loss_function: Literal["ctc", "cross_entropy"] = "ctc"


class W2VSUCModel(B2TModel):
    def __init__(self, config: W2VSUCArgsModel):
        super().__init__()
        self.config = config
        w2v_config = cast(
            Wav2Vec2Config,
            Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h"),
        )
        self.w2v_feature_extractor = cast(
            Wav2Vec2WithoutTransformerModel,
            Wav2Vec2WithoutTransformerModel.from_pretrained(
                "facebook/wav2vec2-base-960h", config=w2v_config
            ),
        )
        self.dropout = nn.Dropout(config.suc_dropout)
        self.suc = create_fully_connected(
            768, len(PHONE_DEF_SIL) + 1, config.suc_hidden_sizes
        )
        if self.config.loss_function == "ctc":
            self.loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        elif self.config.loss_function == "cross_entropy":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Only ctc and cross entropy are supported")

    def forward(self, batch: PhonemeSampleBatch | TimitSeqSampleBatch) -> ModelOutput:
        if batch.target is None:
            raise ValueError("Target is required for training")

        device = batch.input.device
        w2v_output = self.w2v_feature_extractor(batch.input)
        w2v_output = self.dropout(w2v_output)
        suc_output = self.suc(w2v_output)

        feature_extract_output_lens = cast(
            torch.LongTensor,
            self.w2v_feature_extractor._get_feat_extract_output_lengths(
                cast(torch.LongTensor, batch.input_lens.long())
            ),
        )

        if type(batch) == PhonemeSampleBatch:
            if batch.target_lens is None:
                raise ValueError("Target is required for training")
            assert type(self.loss) == nn.CTCLoss
            loss = self.loss.forward(
                log_softmax(suc_output, -1).transpose(0, 1),
                batch.target,
                feature_extract_output_lens.to(device),
                batch.target_lens.to(device),
            )
            metrics = {"ctc_loss": loss.item()}
        elif type(batch) == TimitSeqSampleBatch:
            loss = self._crossEntropyLossBatch(suc_output, batch.phonemes)
            metrics = {"sum_cross_entropy_loss": loss.item()}

        return ModelOutput(
            suc_output,
            metrics,
            loss=loss,
            logit_lens=feature_extract_output_lens,
        )

    def _crossEntropyLossBatch(
        self, output: torch.Tensor, target: list[list[Phoneme]]
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, dtype=torch.float32).cuda()
        for i, phonemes in enumerate(target):
            loss += self._crossEntropyLoss(output[i, :, :], phonemes)
        return loss / len(target)

    def _crossEntropyLoss(
        self, output: torch.Tensor, target: list[Phoneme]
    ) -> torch.Tensor:
        assert type(self.loss) == nn.CrossEntropyLoss
        loss = torch.tensor(0.0, dtype=torch.float32).cuda()
        for phoneme in target:
            # This phoneme could not be mapped to our original phoneme dictionary
            if phoneme.id == -1:
                continue

            # Calculating the start and end points in milliseconds
            # Formula: <given_recording_unit> * <time_per_recording_unit> * <conversion_to_ms>
            start_ms = int(phoneme.start * (1 / 16000) * 1000)
            end_ms = int(phoneme.end * (1 / 16000) * 1000)

            # Starting point of the receptive field of any given output is 20 ms * idx_of_output
            # End point of the receptive field of any given output is 25ms + 20 ms * idx_of_output
            start = math.ceil(start_ms / 20)
            end = max(start + 1, math.floor((end_ms - 25) / 20))

            if end > output.size(0):
                continue

            window = output[start:end, :]
            target_tensor = torch.full(
                (end - start,), phoneme.id, dtype=torch.long
            ).cuda()
            # TODO: properly use CrossEntropyLoss (expects softmax input) + we can maybe avoid manual batching
            phoneme_loss = self.loss.forward(window, target_tensor)

            loss += phoneme_loss
        return loss / len(target)


class Wav2Vec2WithoutTransformerModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        hidden_states, extract_features = self.feature_projection(extract_features)
        return hidden_states
