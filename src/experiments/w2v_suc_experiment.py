import os
from torch.optim.optimizer import Optimizer
from src.args.base_args import BaseExperimentArgsModel
from src.model.audio_wav2vec_model import AudioWav2VecModel
from src.experiments.experiment import Experiment
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, cast
from src.args.wav2vec_args import AudioWav2VecArgsModel
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from src.train.evaluator import DefaultEvaluator


class W2VSUCArgsModel(BaseExperimentArgsModel):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h", "facebook/wav2vec2-base-960h"
    ] = "facebook/wav2vec2-base-960h"
    unfreeze_strategy: Literal["wav2vec2featureextractor", "all"] = "all"
    tokenizer: Literal["wav2vec_pretrained"] = "wav2vec_pretrained"
    remove_punctuation: bool = True


class W2VSUCExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        base_dir = os.path.join(yamlConfig.cache_dir, "audio")
        cache_dir = os.path.join(base_dir, "cache")
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        self._hugg_dataset = load_dataset(
            "google/fleurs", name="en_us", cache_dir=cache_dir, data_dir=data_dir
        )
        self.config = AudioWav2VecArgsModel(**config)
        self.tokenizer = cast(PreTrainedTokenizer, self._create_tokenizer())
        super().__init__(config, yamlConfig)
        self.model: AudioWav2VecModel = self.model

    def get_name(self) -> str:
        return "w2v_suc"

    @staticmethod
    def get_args_model():
        return W2VSUCArgsModel

    def _create_tokenizer(self):
        if self.config.tokenizer == "wav2vec_pretrained":
            return AutoTokenizer.from_pretrained(
                self.config.wav2vec_checkpoint,
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
        raise NotImplementedError()
        return model

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "wav2vec2featureextractor":
                return [
                    {
                        "params": self.model.wav2vec2.wav2vec2.feature_extractor.parameters()
                    },
                ]
            if self.config.unfreeze_strategy == "all":
                return self.model.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for wav2vec experiment"
            )

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)

    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        raise NotImplementedError()

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        ds = self._create_dataset(split)
        return DataLoader(
            self._create_dataset(split),
            batch_size=self.base_config.batch_size,
            shuffle=True,
            collate_fn=ds.get_collate_fn(),
        )

    def get_vocab(self) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(
            list(range(self.tokenizer.vocab_size))
        )

    def create_evaluator(self, mode: Literal["train", "val", "test"]):
        return DefaultEvaluator(self.tokenizer, mode)
