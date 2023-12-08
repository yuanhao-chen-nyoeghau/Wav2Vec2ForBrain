from src.model.b2tmodel import B2TModel, ModelOutput
from transformers import Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
from src.args.wav2vec_args import Wav2VecArgsModel
from torch.nn import Linear, Sequential, ReLU
import torch
from typing import Optional
from src.args.yaml_config import YamlConfigModel
from transformers import PreTrainedTokenizer

# brain data is binned in 20 ms bins (50 Hz)
# wav2vec2 was trained on 16kHz audio
# 16000 Hz / 50 Hz = 320
# --> for each 20 ms bin, we need 320 scalars
# naive approach: upsample bins via shared fully connected NN ("shared_fc")
# (Problem: spreads informaton of a single timestamp across the input)


class B2TWav2Vec(B2TModel):
    def __init__(
        self,
        config: Wav2VecArgsModel,
        yaml_config: YamlConfigModel,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.config = config

        self.brain2audioshape = self._get_brain2audioshape_module()
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(
            config.wav2vec_checkpoint, cache_dir=yaml_config.cache_dir
        )
        self.tokenizer = tokenizer

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        assert (
            len(x.size()) == 2 or len(x.size()) == 3
        ), "x must be 2D shape: (timestamps, brain_data) or 3D shape: (batch_size, timestamps, brain_data)"

        is_batched = len(x.size()) == 3

        batched_input = x if is_batched else x.unsqueeze(0)
        result = []
        for t in range(batched_input.size(1)):
            timestamp_batch = batched_input[:, t, :]
            result.append(self.brain2audioshape(timestamp_batch))

        # result shape: (timestamps, batch_size, 320)
        # audio shaped data: (batch_size, timestamps * 320)

        # Example (batch_size=3, timestamps=2, 320=2):
        # >>> t1 = torch.tensor([[1,2],[3,4],[5,6]])
        # >>> t2 = torch.tensor([[7,8],[9,10],[11,12]])
        # >>> torch.stack([t1,t2],dim=1)
        # tensor([[[ 1,  2],
        # [ 7,  8]],
        #
        # [[ 3,  4],
        # [ 9, 10]],
        #
        # [[ 5,  6],
        # [11, 12]]])
        audio_shaped_data = torch.stack(result, dim=1).flatten(start_dim=1)

        # replace padding token with -100 for it to be ignored in ctc loss
        if targets is not None:
            targets = torch.where(
                targets == self.tokenizer.pad_token_id, torch.tensor(-100), targets
            )

        wav2vec2_out: CausalLMOutput = self.wav2vec2(
            audio_shaped_data, return_dict=True, labels=targets
        )
        return ModelOutput(
            logits=wav2vec2_out.logits,
            loss=wav2vec2_out.loss,
        )

    def _get_brain2audioshape_module(self):
        in_size = self._get_timestamp_vec_len()
        if self.config.brain2audioshape_strategy == "shared_fc":
            return Linear(in_size, 320)
        elif self.config.brain2audioshape_strategy == "shared_fc_relu":
            return Sequential(Linear(in_size, 320), ReLU())

        raise Exception(
            f"Unknown brain2audioshape strategy {self.config.brain2audioshape_strategy}"
        )

    def _get_timestamp_vec_len(self):
        if self.config.preprocessing == "competition_recommended":
            return 256
        elif self.config.preprocessing == "seperate_zscoring":
            return 256
        elif self.config.preprocessing == "only_tx_unnormalized":
            return 128
        elif self.config.preprocessing == "only_tx_zscored":
            return 128
        elif self.config.preprocessing == "only_spikepow_unnormalized":
            return 128
        elif self.config.preprocessing == "only_spikepow_zscored":
            return 128
        raise Exception(f"Unknown preprocessing type {self.config.preprocessing}")
