from src.datasets.brain2text import Brain2TextDataset
from src.experiments.b2t_experiment import B2TArgsModel
from src.experiments.b2t_experiment import B2TExperiment
from src.args.b2t_audio_args import B2TAudioDatasetArgsModel
from src.model.b2tmodel import B2TModel
from src.model.mvts_transformer_model import (
    B2TMvtsTransformerArgsModel,
    MvtsTransformerModel,
)
from src.args.yaml_config import YamlConfigModel
from typing import Literal
from torch.utils.data import Dataset


class B2P2TMvtsArgsModel(B2TMvtsTransformerArgsModel, B2TArgsModel):
    pass


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
        return B2P2TMvtsArgsModel

    def _create_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        model = MvtsTransformerModel(self.config, vocab_size=1, in_size=5)
        return model

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> Dataset:
        return Brain2TextDataset(
            config=self.ds_config,
            yaml_config=self.yaml_config,
            split=split,
        )
