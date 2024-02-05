from src.experiments.b2t_experiment import B2TExperiment
from src.args.b2t_audio_args import B2TAudioDatasetArgsModel
from src.datasets.b2t_audio import B2TAudioDataset
from src.model.b2tmodel import B2TModel
from src.model.mvts_transformer import TSTransformerEncoderClassiregressor
from src.args.yaml_config import YamlConfigModel
from typing import Literal
from torch.utils.data import Dataset


class B2TMvtsTransformerArgsModel(B2TArgsModel):
    hidden_size: int = 256
    bidirectional: bool = True
    num_gru_layers: int = 2
    bias: bool = True
    dropout: float = 0.0
    learnable_inital_state: bool = False
    classifier_hidden_sizes: list[int] = [256, 128, 64]
    classifier_activation: Literal["gelu"] = "gelu"


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
        return "gru_experiment"

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
        model = TSTransformerEncoderClassiregressor(
            feat_dim=192,
            d_model=192,
            max_len=201,
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
        return model
    
    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> Dataset:
        return B2TAudioDataset(
            config=self.ds_config,
            yaml_config=self.yaml_config,
            split=split,
        )