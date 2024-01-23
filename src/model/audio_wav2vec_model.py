from src.model.b2tmodel import B2TModel, ModelOutput
from transformers import Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
from src.args.wav2vec_args import AudioWav2VecArgsModel
from torch.nn import Linear, Sequential, ReLU, Flatten, Unflatten
import torch
from typing import Optional, cast
from src.args.yaml_config import YamlConfigModel
from transformers import PreTrainedTokenizer


class AudioWav2VecModel(B2TModel):
    def __init__(
        self,
        config: AudioWav2VecArgsModel,
        yaml_config: YamlConfigModel,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.config = config

        self.wav2vec2 = cast(
            Wav2Vec2ForCTC,
            Wav2Vec2ForCTC.from_pretrained(
                config.wav2vec_checkpoint,
                cache_dir=yaml_config.cache_dir,
                ctc_loss_reduction=config.ctc_loss_reduction,
            ),
        )
        print("config", self.wav2vec2.config)
        self.tokenizer = tokenizer

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        assert (
            len(x.size()) == 1 or len(x.size()) == 2
        ), "x must be 1D shape: (, audio_data) or 3D shape: (batch_size, audio_data)"

        is_batched = len(x.size()) == 2

        batched_input = x if is_batched else x.unsqueeze(0)

        # replace padding token with -100 for it to be ignored in ctc loss
        if targets is not None:
            targets = torch.where(
                targets == self.tokenizer.pad_token_id, torch.tensor(-100), targets
            )

        wav2vec2_out: CausalLMOutput = self.wav2vec2(
            batched_input, return_dict=True, labels=targets
        )
        metrics = (
            {"ctc_loss": wav2vec2_out.loss.item()}
            if wav2vec2_out.loss is not None
            else {}
        )

        return ModelOutput(
            logits=wav2vec2_out.logits,
            loss=wav2vec2_out.loss,
            metrics=metrics,
        )
