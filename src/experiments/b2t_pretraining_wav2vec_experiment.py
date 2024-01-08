from src.model.b2t_pretraining_wav2vec_model import B2TPretrainingWav2VecModel
from src.experiments.b2t_pretraining_wav2vec_experiment import (
    B2TPretrainingWav2VecModel,
)
from src.experiments.b2t_wav2vec_shared_agg_experiment import (
    B2TWav2VecSharedAggregationExperiment,
)
from src.args.yaml_config import YamlConfigModel
from src.args.wav2vec_args import B2TWav2VecSharedAggregationArgsModel
from src.model.b2tmodel import B2TModel
from src.train.history import TrainHistory
import os


class B2TWav2VecPretrainingExperiment(B2TWav2VecSharedAggregationExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        super().__init__(config, yamlConfig)
        self.config = B2TWav2VecSharedAggregationArgsModel(**config)

        self.model: B2TPretrainingWav2VecModel = self.model

    def get_name(self) -> str:
        return "b2t_wav2vec_pretraining"

    @staticmethod
    def get_args_model():
        return B2TWav2VecSharedAggregationArgsModel

    def _create_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "contrastive_loss",  # type: ignore
            "Only contrastive_loss loss is currently supported",
        )
        model = B2TPretrainingWav2VecModel(
            config=self.config,
            yaml_config=self.yaml_config,
            tokenizer=self.tokenizer,
        )
        return model

    def run_real_world_test(self, model: B2TModel):
        pass

    def plot_results(self, history: TrainHistory):
        super().plot_results(history)

        test_cosine_similarities = [
            item.metrics["cosine_similarity"] for item in history.test_losses.metrics
        ]

        import matplotlib.pyplot as plt

        # Sample data: Replace these with your actual float arrays

        # Create a figure and a set of subplots
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # Histogram for data1
        ax.hist(test_cosine_similarities, bins=10, color="blue", alpha=0.7)
        ax.set_title("Test Set")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Frequency")

        # Display the histograms
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "histogram.png"))
