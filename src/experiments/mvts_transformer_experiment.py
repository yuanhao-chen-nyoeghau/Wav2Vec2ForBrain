import torch
from src.datasets.brain2text import Brain2TextDataset
from src.args.base_args import B2TArgsModel
from src.experiments.b2t_experiment import B2TExperiment
from src.args.b2t_audio_args import B2TAudioDatasetArgsModel
from src.model.b2tmodel import B2TModel, ModelOutput
from src.model.mvts_transformer_model import TSTransformerEncoderClassiregressor
from src.args.yaml_config import YamlConfigModel
from typing import Literal, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from torch.nn.functional import log_softmax
from torch import nn

def calc_seq_len(index_seq: torch.Tensor):
    for i in range(len(index_seq)):
        j = len(index_seq) - 1 - i
        if index_seq[j].item() > 0:
            return j + 1
    return 0

class B2TMvtsTransformerArgsModel(B2TArgsModel):
    hidden_size: int = 256
    bidirectional: bool = True
    num_gru_layers: int = 2
    bias: bool = True
    dropout: float = 0.0
    learnable_inital_state: bool = False
    classifier_hidden_sizes: list[int] = [256, 128, 64]
    classifier_activation: Literal["gelu"] = "gelu"

class MvtsTransformerModel(B2TModel):
    def __init__(self, config: B2TMvtsTransformerArgsModel, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = 950
        self.loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.transformer = TSTransformerEncoderClassiregressor(
            feat_dim=256,
            d_model=256,
            max_len=self.max_len,
            n_heads=2,
            num_layers=2,
            dim_feedforward=256,
            num_classes=31,
            dropout=0,
            pos_encoding="learnable",
            activation="gelu",
            norm="BatchNorm",
            freeze=False,
        )

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        assert targets is not None, "Targets must be set"
        device = targets.device
        target_lens = torch.tensor([calc_seq_len(seq) for seq in targets])
        if targets is not None:
            targets = torch.where(
                targets == self.tokenizer.pad_token_id, torch.tensor(-100), targets
            )
        x_padded = torch.zeros(x.shape[0] , self.max_len , x.shape[2])
        x_padded[:, :x.shape[1], :] = x
        target_lens = torch.tensor([calc_seq_len(seq) for seq in targets])
        mask = x_padded != 0
        sequence_mask = mask.any(-1)
        out = self.transformer.forward(x_padded.to(device), sequence_mask.to(device))
        out = log_softmax(out, -1)

        # in_seq_lens shape: (batch_size, seq_len)
        in_seq_lens = sequence_mask.sum(-1)
        # in_seq_lens shape: (batch_size)
        in_seq_lens = in_seq_lens.clamp(max=out.shape[1])   
        out = out.transpose(0, 1)
        # out shape: (seq_len, batch_size, vocab_size)

        ctc_loss = self.loss(
            out,
            targets,
            in_seq_lens.to(device),
            target_lens.to(device),
        )

        if ctc_loss.item() < 0:
            print(
                f"\nWarning: loss is negative, this might be due to prediction lens ({in_seq_lens.tolist()}) being smaller than target lens {target_lens.tolist()}\n"
            )

        return ModelOutput(
            out.transpose(0, 1),
            {"ctc_loss": ctc_loss.item()},
            ctc_loss,
        )



class MvtsTransformerExperiment(B2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        self.ds_config = B2TAudioDatasetArgsModel(**config)
        super().__init__(config, yamlConfig)
        self.model: B2TModel = self.model

        assert (
            self.config.preprocessing == "seperate_zscoring"
        ), "Only seperate_zscoring is currently supported"

    def get_name(self) -> str:
        return "mvts_transformer_experiment"

    @staticmethod
    def get_args_model():
        return B2TMvtsTransformerArgsModel

    def _create_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        model = MvtsTransformerModel(self.config, self.tokenizer)
        return model
    
    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> Dataset:
        return Brain2TextDataset(
            config=self.ds_config,
            yaml_config=self.yaml_config,
            split=split,
        )