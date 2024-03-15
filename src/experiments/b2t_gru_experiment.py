from typing import cast
from src.args.base_args import B2TArgsModel
from src.model.gru_model import GRUModel, GruArgsModel
from src.experiments.b2t_experiment import B2TExperiment
from src.model.b2tmodel import B2TModel
from src.args.yaml_config import YamlConfigModel


class B2tGruArgsModel(GruArgsModel, B2TArgsModel):
    pass


class B2tGruExperiment(B2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)
        self.model: B2TModel = self.model

        assert (
            self.config.preprocessing == "seperate_zscoring"
        ), "Only seperate_zscoring is currently supported"

    def get_name(self) -> str:
        return "gru_experiment"

    @staticmethod
    def get_args_model():
        return B2tGruArgsModel

    def _create_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        model = GRUModel(
            self.config,
            self.tokenizer.vocab_size,
            256,
            cast(int, self.tokenizer.pad_token_id),
        )
        return model
