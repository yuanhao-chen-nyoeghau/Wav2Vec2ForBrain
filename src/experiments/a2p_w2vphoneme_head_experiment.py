import os
import torch
from torch.optim.optimizer import Optimizer
from src.experiments.suc_approach.A__timit_w2v_suc_ctc_experiment import (
    TimitSeqW2VSUCEvaluator,
)
from src.model.w2vphoneme_head import (
    W2VPhonemeHead,
    W2VPhonemeHeadArgs,
    W2VPhonemeWithCustomHead,
)
from src.experiments.experiment import Experiment
from src.datasets.timit_a2p_seq_dataset import TimitA2PSeqDataset
from src.datasets.audio_with_phonemes_seq import (
    AudioWPhonemesDatasetArgsModel,
)
from src.util.phoneme_helper import PHONE_DEF_SIL
from src.args.base_args import BaseExperimentArgsModel
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, cast
from torch.utils.data import DataLoader


class A2P_W2VPhonemeHeadExperimentArgs(
    BaseExperimentArgsModel, AudioWPhonemesDatasetArgsModel, W2VPhonemeHeadArgs
):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal["facebook/wav2vec2-lv-60-espeak-cv-ft"] = (
        "facebook/wav2vec2-lv-60-espeak-cv-ft"
    )
    unfreeze_strategy: Literal["head"] = "head"


class A2P_W2VPhonemeHeadExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "a2p_w2vphoneme_head"

    @staticmethod
    def get_args_model():
        return A2P_W2VPhonemeHeadExperimentArgs

    def _create_model(self):
        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is supported",
        )
        head = W2VPhonemeHead(self.config, out_size=len(self.get_vocab()))
        model = W2VPhonemeWithCustomHead(self.config.wav2vec_checkpoint, head)
        return model

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "head":
                return cast(W2VPhonemeWithCustomHead, self.model).head.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for {self.get_name()} experiment"
            )

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)

    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        return TimitA2PSeqDataset(self.config, self.yaml_config, split=split)

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        ds = self._create_dataset(split)
        return DataLoader(
            self._create_dataset(split),
            batch_size=self.base_config.batch_size,
            shuffle=True,
            collate_fn=ds.get_collate_fn(),
        )

    def get_vocab(self) -> list[str]:
        return ["BLANK"] + PHONE_DEF_SIL

    def create_evaluator(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        return TimitSeqW2VSUCEvaluator(mode, track_non_test_predictions)

    def store_trained_model(self, trained_model: W2VPhonemeWithCustomHead):
        torch.save(
            trained_model.head.state_dict(),
            os.path.join(self.results_dir, "head.pt"),
        )
