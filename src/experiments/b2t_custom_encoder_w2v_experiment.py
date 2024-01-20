from src.model.b2t_custom_encoder_w2v_model import (
    B2TCustomEncoderW2VFineTuningModel,
    B2TCustomEncoderW2VPretrainingModel,
)
from src.experiments.b2t_wav2vec_experiment import B2TWav2VecExperiment
from src.args.yaml_config import YamlConfigModel
from src.args.wav2vec_args import (
    B2TWav2VecCustomEncoderArgsModel,
)
from src.train.history import TrainHistory
import os


class B2TWav2VecCustomEncoderExperiment(B2TWav2VecExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        super().__init__(config, yamlConfig)
        self.config = B2TWav2VecCustomEncoderArgsModel(**config)
        self.model: B2TCustomEncoderW2VPretrainingModel | B2TCustomEncoderW2VFineTuningModel = (
            self.model
        )

    def get_name(self) -> str:
        return "b2t_wav2vec_custom_encoder"

    @staticmethod
    def get_args_model():
        return B2TWav2VecCustomEncoderArgsModel

    def _create_model(self):
        if self.config.mode == "pretraining":
            return B2TCustomEncoderW2VPretrainingModel(
                self.config, self.yaml_config, self.tokenizer
            )
        elif self.config.mode == "finetuning":
            return B2TCustomEncoderW2VFineTuningModel(
                self.config, self.yaml_config, self.tokenizer
            )
        raise Exception(
            f"Mode {self.config.mode} not supported yet by {self.get_name()}"
        )

    def plot_results(self, history: TrainHistory):
        super().plot_results(history)

        if self.config.mode != "pretraining":
            return

        out_dir = os.path.join(self.results_dir, "histograms")

        history.plot_metric_histograms(out_dir, "cosine_similarity")
