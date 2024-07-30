from torch.optim.optimizer import Optimizer
from src.datasets.brain2text import Brain2TextDataset
from src.experiments.experiment import Experiment
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, cast
from src.args.wav2vec_args import B2TWav2VecArgsModel
from transformers import AutoTokenizer
from src.model.b2t_wav2vec_model import B2TWav2Vec
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.train.evaluator import DefaultEvaluator


class B2TWav2VecExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)
        self.tokenizer = cast(PreTrainedTokenizer, self._create_tokenizer())

        self.model: B2TWav2Vec = self.model

    def get_name(self) -> str:
        return "b2t_wav2vec"

    @staticmethod
    def get_args_model():
        return B2TWav2VecArgsModel

    def _create_tokenizer(self):
        if self.config.tokenizer == "wav2vec_pretrained":
            assert (
                not self.config.tokenizer_checkpoint is None
            ), "Tokenizer checkpoint (--tokenizer_checkpoint) must be set when using --tokenizer=wav2vec_pretrained"

            return AutoTokenizer.from_pretrained(
                self.config.tokenizer_checkpoint,
                cache_dir=self.yaml_config.cache_dir,
            )
        raise Exception(f"Tokenizer {self.config.tokenizer} not supported yet")

    def _create_model(self):
        assert (
            self.config.tokenizer == "wav2vec_pretrained"
        ), "Only pretrained wav2vec is currently supported"

        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is currently supported",
        )
        model = B2TWav2Vec(
            config=self.config,
            yaml_config=self.yaml_config,
            tokenizer=self.tokenizer,
        )
        return model

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "wav2vec2featureextractor_ours":
                return [
                    {"params": self.model.brain2audioshape.parameters()},
                    {
                        "params": self.model.wav2vec2.wav2vec2.feature_extractor.parameters()
                    },
                ]
            if (
                self.config.unfreeze_strategy
                == "wav2vec2featureextractor_wav2vec2classifier_ours"
            ):
                return [
                    {"params": self.model.brain2audioshape.parameters()},
                    {
                        "params": self.model.wav2vec2.wav2vec2.feature_extractor.parameters()
                    },
                    {"params": self.model.wav2vec2.wav2vec2.head.parameters()},
                ]
            if self.config.unfreeze_strategy == "lm_head":
                return self.model.wav2vec2.lm_head.parameters()
            if self.config.unfreeze_strategy == "all":
                return self.model.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for wav2vec experiment"
            )

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> Dataset:
        return Brain2TextDataset(
            config=self.config,
            yaml_config=self.yaml_config,
            split=split,
            tokenizer=self.tokenizer,
        )

    def get_vocab(self) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(
            list(range(self.tokenizer.vocab_size))
        )

    def create_evaluator(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        return DefaultEvaluator(self.tokenizer, mode, track_non_test_predictions)
