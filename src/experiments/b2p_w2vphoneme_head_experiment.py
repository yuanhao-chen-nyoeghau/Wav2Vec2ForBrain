import os
from pydantic import Field
import torch
from torch.optim.optimizer import Optimizer
from src.model.w2v_custom_feat_extractor import (
    W2VBrainEncoderModel,
    W2VBrainEncoderModelArgs,
)
from src.datasets.discriminator_dataset import (
    B2P2TBrainFeatureExtractorArgsModel,
    DiscriminatorDataset,
)
from src.experiments.suc_approach.B__b2p_suc_experiment import B2PSUCEvaluator
from src.datasets.brain2text_w_phonemes import Brain2TextWPhonemesDataset
from src.model.w2vphoneme_head import (
    W2VPhonemeHead,
    W2VPhonemeHeadArgs,
)
from src.experiments.experiment import Experiment
from src.util.phoneme_helper import PHONE_DEF_SIL
from src.args.base_args import B2TDatasetArgsModel, BaseExperimentArgsModel
from src.args.yaml_config import YamlConfigModel
from typing import Any, Literal, Optional, cast
from torch.utils.data import DataLoader

from src.util.warmup_scheduler import get_2module_warmup_scheduler


class B2P_W2VPhonemeHeadExperimentArgs(
    BaseExperimentArgsModel,
    B2TDatasetArgsModel,
    W2VPhonemeHeadArgs,
    B2P2TBrainFeatureExtractorArgsModel,
    W2VBrainEncoderModelArgs,
):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal["facebook/wav2vec2-lv-60-espeak-cv-ft"] = (
        "facebook/wav2vec2-lv-60-espeak-cv-ft"
    )
    unfreeze_strategy: Literal["brain_encoder", "brain_encoder+w2v"] = "brain_encoder"
    w2v_learning_rate: Optional[float] = None
    w2v_warmup_start_step: Optional[int] = Field(
        default=None,
        description="Epoch at which warm up phase of w2v lr starts. Before LR will be 0. 0 if not provided",
    )
    w2v_warmup_steps: Optional[int] = Field(
        default=None,
        description="Num epochs from w2v_warmup_start_step to reach full w2v_learning_rate. 0 if not provided",
    )
    adjust_global_lr_to_w2v_postwarmup_lr: Optional[bool] = Field(
        description="Adjust the global learning rate to that of w2v over w2v warmup interval, then keep at w2v_learning_rate. Only valid when brain_encoder+w2v unfreeze strategy is set."
    )
    head_checkpoint: Optional[str] = None
    brain_encoder_checkpoint: Optional[str] = None


class B2P_W2VPhonemeHeadExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "b2p_w2vphoneme_head"

    @staticmethod
    def get_args_model():
        return B2P_W2VPhonemeHeadExperimentArgs

    def _create_model(self):
        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is supported",
        )
        head = W2VPhonemeHead(self.config, out_size=len(self.get_vocab()))
        if self.config.head_checkpoint is not None:
            head.load_state_dict(torch.load(self.config.head_checkpoint))
        neural_decoder = DiscriminatorDataset.brain_feature_extractor_from_config(
            self.config,
            self.config.brain_encoder_checkpoint,
            self.config.wav2vec_checkpoint,
        )
        model = W2VBrainEncoderModel(
            config=self.config,
            brain_encoder=neural_decoder,
            wav2vec_checkpoint=self.config.wav2vec_checkpoint,
            head=head,
        )
        return model

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "brain_encoder+w2v":
                return [
                    {
                        "params": cast(
                            W2VBrainEncoderModel, self.model
                        ).brain_encoder.parameters()
                    },
                    {
                        "params": (
                            list(
                                cast(
                                    W2VBrainEncoderModel, self.model
                                ).w2v_encoder.parameters()
                            )
                            + (
                                list(
                                    cast(
                                        W2VBrainEncoderModel, self.model
                                    ).head.parameters()
                                )
                                if cast(W2VBrainEncoderModel, self.model).head
                                is not None
                                else []
                            )
                        ),
                        "lr": (
                            self.config.w2v_learning_rate
                            if self.config.w2v_learning_rate is not None
                            else self.config.learning_rate
                        ),
                    },
                ]
            if self.config.unfreeze_strategy == "brain_encoder":
                assert (
                    self.config.w2v_learning_rate is None
                ), "w2v_learning_rate can only be set if unfreeze strategy is brain_encoder+w2v"
                return cast(W2VBrainEncoderModel, self.model).brain_encoder.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for wav2vec experiment"
            )

        optim_cls: Any = self._get_optimizer_cls()

        return optim_cls(
            get_trainable_params(),
            lr=self.base_config.learning_rate,
            weight_decay=self.base_config.weight_decay,
            eps=self.base_config.optimizer_epsilon,
        )

    def get_scheduler(
        self, optimizer: Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        if self.config.unfreeze_strategy == "brain_encoder":
            # return scheduler from super
            assert (
                self.config.w2v_warmup_steps is None
            ), "w2v_warmup_steps can only be set if unfreeze strategy is brain_encoder+w2v"
            assert (
                self.config.adjust_global_lr_to_w2v_postwarmup_lr is None
            ), "adjust_global_lr_to_w2v_postwarmup_lr can only be set if unfreeze strategy is brain_encoder+w2v"
            return super().get_scheduler(optimizer)

        warmup_start_step = (
            self.config.w2v_warmup_start_step
            if self.config.w2v_warmup_start_step
            else 0
        )
        warmup_steps = (
            self.config.w2v_warmup_steps if self.config.w2v_warmup_steps else 0
        )

        return get_2module_warmup_scheduler(
            optimizer,
            self.config.learning_rate,
            warmup_start_step,
            warmup_steps,
            (
                self.config.w2v_learning_rate
                if self.config.w2v_learning_rate is not None
                else self.config.learning_rate
            ),
            self.config.adjust_global_lr_to_w2v_postwarmup_lr == True,
        )

    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        return Brain2TextWPhonemesDataset(self.config, self.yaml_config, split=split)

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
        return B2PSUCEvaluator(mode, track_non_test_predictions)

    def store_trained_model(self, trained_model: W2VBrainEncoderModel):
        torch.save(
            trained_model.state_dict(),
            os.path.join(self.results_dir, "model.pt"),
        )
