from args.wav2vec_args import B2TAudioWav2VecArgsModel
from args.yaml_config import YamlConfigModel
from src.model.b2tmodel import B2TModel, ModelOutput
from transformers import PreTrainedTokenizer
from typing import Optional, cast
from transformers import Wav2Vec2ForCTC
import torch
from transformers.modeling_outputs import CausalLMOutput


class B2TAudioWav2VecModel(B2TModel):
    def __init__(
        self,
        config: B2TAudioWav2VecArgsModel,
        yaml_config: YamlConfigModel,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
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
        if not self.config.mean_reduction:
            self.summarizer_module = torch.nn.Sequential(
                torch.nn.Linear(128, self.config.hidden_nodes),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config.hidden_nodes, 1),
            )

        print("config", self.wav2vec2.config)
        self.tokenizer = tokenizer

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        is_batched = len(x.size()) > 1

        batched_input = x if is_batched else x.unsqueeze(0)

        # replace padding token with -100 for it to be ignored in ctc loss
        if targets is not None:
            targets = torch.where(
                targets == self.tokenizer.pad_token_id, torch.tensor(-100), targets
            )

        summarizer_out = self.summarizer_module(batched_input)
        summarizer_out = summarizer_out.squeeze(-1)
        model_out: CausalLMOutput = self.wav2vec2(
            summarizer_out, return_dict=True, labels=targets
        )

        return ModelOutput(
            logits=model_out.logits,
            loss=model_out.loss,
        )
