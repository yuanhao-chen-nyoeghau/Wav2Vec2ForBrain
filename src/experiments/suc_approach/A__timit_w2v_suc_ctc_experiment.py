import os
import torch
from torch.optim.optimizer import Optimizer
from src.experiments.experiment import Experiment
from src.datasets.timit_a2p_seq_dataset import TimitA2PSeqDataset
from src.datasets.audio_with_phonemes_seq import (
    AudioWPhonemesDatasetArgsModel,
)
from src.util.phoneme_helper import PHONE_DEF_SIL
from src.model.w2v_suc_ctc_model import W2VSUC_CTCArgsModel, W2VSUCForCtcModel
from src.args.base_args import BaseExperimentArgsModel
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, Optional, cast
from torch.utils.data import DataLoader
from src.train.evaluator import TimitSeqW2VSUCEvaluator


class W2VSUCCtcExperimentArgsModel(
    BaseExperimentArgsModel, W2VSUC_CTCArgsModel, AudioWPhonemesDatasetArgsModel
):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal[
        "facebook/wav2vec2-base-100h", "facebook/wav2vec2-base-960h"
    ] = "facebook/wav2vec2-base-960h"
    unfreeze_strategy: Literal["suc", "suc+gru"] = "suc"
    suc_checkpoint: Optional[str] = None


class TimitW2VSUC_CTCExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "timit_w2v_suc_ctc"

    @staticmethod
    def get_args_model():
        return W2VSUCCtcExperimentArgsModel

    def _create_model(self):
        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is supported",
        )
        model = W2VSUCForCtcModel(self.config)
        if self.config.suc_checkpoint is not None:
            model.suc_for_ctc.suc.load_state_dict(
                torch.load(self.config.suc_checkpoint, map_location="cuda"),
                strict=True,
            )
        return model

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "suc":
                return cast(
                    W2VSUCForCtcModel, self.model
                ).suc_for_ctc.ctc_head.parameters()
            elif self.config.unfreeze_strategy == "suc+gru":
                return cast(W2VSUCForCtcModel, self.model).suc_for_ctc.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for wav2vec experiment"
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

    def store_trained_model(self, trained_model: W2VSUCForCtcModel):
        torch.save(
            trained_model.suc_for_ctc.state_dict(),
            os.path.join(self.results_dir, "suc_for_ctc.pt"),
        )
