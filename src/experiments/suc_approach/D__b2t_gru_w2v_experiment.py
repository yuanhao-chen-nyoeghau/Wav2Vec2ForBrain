from typing import Any, Literal, cast

from git import Optional
from pydantic import Field
import torch
from src.experiments.b2t_experiment import B2TArgsModel, B2TExperiment
from src.datasets.discriminator_dataset import (
    B2P2TBrainFeatureExtractorArgsModel,
    DiscriminatorDataset,
)
from src.model.w2v_custom_feat_extractor import (
    W2VBrainEncoderModel,
    W2VBrainEncoderModelArgs,
)
from src.args.yaml_config import YamlConfigModel
from torch.optim.optimizer import Optimizer

# Baseline Experiment: b2p2t_gru:
# 5gram + rescoring: 0.28 WER (/hpi/fs00/scratch/tobias.fiedler/brain2text/experiment_results/b2p2t_gru/2024-03-27_17#31#07),
# 3gram: 0.3153 WER
# (/hpi/fs00/scratch/tobias.fiedler/brain2text/experiment_results/b2p2t_gru/2024-03-28_08#18#39)


class B2TGruAndW2VArgsModel(
    B2TArgsModel, B2P2TBrainFeatureExtractorArgsModel, W2VBrainEncoderModelArgs
):
    brain_encoder_path: Optional[str] = None
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


class B2TGruAndW2VExperiment(B2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "b2p2t_gru+w2v"

    @staticmethod
    def get_args_model():
        return B2TGruAndW2VArgsModel

    def _create_model(self):
        brain_encoder = DiscriminatorDataset.brain_feature_extractor_from_config(
            self.config, self.config.brain_encoder_path, self.config.wav2vec_checkpoint
        )
        model = W2VBrainEncoderModel(
            self.config, brain_encoder, self.config.wav2vec_checkpoint
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
                        "params": cast(
                            W2VBrainEncoderModel, self.model
                        ).w2v_encoder.parameters(),
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
            return super().get_scheduler(optimizer)

        warmup_start_step = (
            self.config.w2v_warmup_start_step
            if self.config.w2v_warmup_start_step
            else 0
        )
        warmup_steps = (
            self.config.w2v_warmup_steps if self.config.w2v_warmup_steps else 0
        )

        from torch.optim.lr_scheduler import LambdaLR

        def custom_lr_lambda(step):
            if step < warmup_start_step:
                return 0.0

            return min(
                1.0,
                (step - warmup_start_step) / warmup_steps if warmup_steps > 0 else 1.0,
            )

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=[
                lambda step: 1.0,  # For brain_encoder parameters, keep lr constant
                custom_lr_lambda,  # For w2v_encoder parameters, apply the custom warmup logic
            ],
            verbose=True,
        )

        return scheduler
