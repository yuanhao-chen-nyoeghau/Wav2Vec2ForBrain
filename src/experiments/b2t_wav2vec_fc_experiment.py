from src.experiments.b2t_wav2vec_experiment import B2TWav2VecExperiment
from src.args.yaml_config import YamlConfigModel
from src.args.wav2vec_args import B2TWav2VecCnnArgsModel, B2TWav2VecFCArgsModel
from src.model.b2t_wav2vec_model import B2TWav2Vec


class B2TWav2VecFCExperiment(B2TWav2VecExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        super().__init__(config, yamlConfig)
        self.config = B2TWav2VecFCArgsModel(**config)

        self.model: B2TWav2Vec = self.model

    def get_name(self) -> str:
        return "b2t_wav2vec_fc"

    @staticmethod
    def get_args_model():
        return B2TWav2VecFCArgsModel
