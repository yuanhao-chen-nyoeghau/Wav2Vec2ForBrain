from torch.optim.optimizer import Optimizer
from src.datasets.brain2text_w_phonemes import (
    Brain2TextWPhonemesDataset,
    PhonemeSampleBatch,
    decode_predicted_phoneme_ids,
)
from src.model.b2tmodel import ModelOutput
from src.args.base_args import (
    B2TArgsModel,
    B2TDatasetArgsModel,
    BaseExperimentArgsModel,
)
from src.experiments.experiment import DecodedPredictionBatch, Experiment
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


class B2P2TArgsModel(BaseExperimentArgsModel, B2TDatasetArgsModel):
    pass


class B2P2TExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def get_args_model():
        return B2P2TArgsModel

    def _create_model(self):
        raise NotImplementedError()

    def decode_predictions(
        self, predictions: ModelOutput, sample: PhonemeSampleBatch
    ) -> DecodedPredictionBatch:
        predicted_ids = predictions.logits.argmax(dim=-1).cpu().numpy()

        # phoneme_ids = sample.target

        # TODO: decode phonemes to strings
        predicted_strings = [
            decode_predicted_phoneme_ids(sample) for sample in predicted_ids
        ]

        label_strings = sample.transcriptions
        return DecodedPredictionBatch(predicted_strings, label_strings)

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            return self.model.parameters()

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)

    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        return Brain2TextWPhonemesDataset(
            config=self.config,
            yaml_config=self.yaml_config,
            split=split,
        )

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        ds = self._create_dataset(split)
        return DataLoader(
            self._create_dataset(split),
            batch_size=self.base_config.batch_size,
            shuffle=True,
            collate_fn=ds.get_collate_fn(),
        )

    def get_vocab(self) -> list[str]:
        return Brain2TextWPhonemesDataset.vocab
