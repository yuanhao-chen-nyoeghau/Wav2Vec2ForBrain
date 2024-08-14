from typing import Any, cast
from git import Optional
from pydantic import Field
from src.datasets.discriminator_dataset import (
    B2P2TBrainFeatureExtractorArgsModel,
    DiscriminatorDataset,
)
from src.model.w2v_custom_feat_extractor import (
    W2VBrainEncoderModel,
    W2VBrainEncoderModelArgs,
)
from src.experiments.b2p2t_experiment import B2P2TArgsModel, B2P2TExperiment
from src.args.yaml_config import YamlConfigModel
from torch.optim.optimizer import Optimizer
from src.util.nn_helper import ACTIVATION_FUNCTION, create_fully_connected
import torch
from src.util.warmup_scheduler import get_2module_warmup_scheduler
from src.util.phoneme_helper import PHONE_DEF_SIL


class B2P2TGruW2vPhonemeArgsModel(
    B2P2TArgsModel, B2P2TBrainFeatureExtractorArgsModel, W2VBrainEncoderModelArgs
):
    brain_encoder_path: Optional[str] = None
    w2v_learning_rate: Optional[float] = None
    w2v_warmup_start_step: Optional[int] = Field(
        default=None,
        description="Epoch at which warm up phase of w2v lr starts. Before LR will be 0. 0 if not provided",
    )
    w2v_warmup_steps: Optional[int] = Field(
        default=None,
        description="Num epochs from w2v_warmup_start_step to reach full w2v_learning_rate. 0 if not provided",
    )
    wav2vec_checkpoint: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    adjust_global_lr_to_w2v_postwarmup_lr: Optional[bool] = Field(
        description="Adjust the global learning rate to that of w2v over w2v warmup interval, then keep at w2v_learning_rate. Only valid when brain_encoder+w2v unfreeze strategy is set."
    )
    head_fc_hidden_sizes: list[int] = []
    head_fc_activation_function: ACTIVATION_FUNCTION = "gelu"
    w2v_skip_loading_weights: bool = Field(
        default=False,
        description="Skip loading weights from wav2vec checkpoint, only load architecture",
    )


class B2P2TGruW2vPhonemeExperiment(B2P2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "b2p2t_gru_w2vphoneme"

    @staticmethod
    def get_args_model():
        return B2P2TGruW2vPhonemeArgsModel

    def _create_neural_decoder(self):
        raise NotImplementedError(
            "Not implemented. Directly use _create_model for this experiment"
        )

    def _create_model(self):
        brain_encoder = DiscriminatorDataset.brain_feature_extractor_from_config(
            self.config, self.config.brain_encoder_path, self.config.wav2vec_checkpoint
        )
        # head to convert from IPA phonemes to ARPABET phonemes
        n_ipa_phonemes = 392
        n_arpabet_phonemes = len(PHONE_DEF_SIL) + 1
        head = create_fully_connected(
            n_ipa_phonemes,
            n_arpabet_phonemes,
            activation=self.config.head_fc_activation_function,
            hidden_sizes=self.config.head_fc_hidden_sizes,
        )

        model = W2VBrainEncoderModel(
            self.config,
            brain_encoder,
            self.config.wav2vec_checkpoint,
            head=head,
            skip_loading_weights=self.config.w2v_skip_loading_weights,
        )
        return model

    def create_optimizer(self) -> Optimizer:
        model = cast(W2VBrainEncoderModel, self.model)

        assert model.head is not None, "Head must be defined for this experiment"
        import itertools

        non_w2v_params = itertools.chain(
            model.brain_encoder.parameters(), model.head.parameters()
        )

        trainable_params = [
            {"params": non_w2v_params},
            {
                "params": model.w2v_encoder.parameters(),
                "lr": (
                    self.config.w2v_learning_rate
                    if self.config.w2v_learning_rate is not None
                    else self.config.learning_rate
                ),
            },
        ]
        optim_cls: Any = self._get_optimizer_cls()

        return optim_cls(
            trainable_params,
            lr=self.base_config.learning_rate,
            weight_decay=self.base_config.weight_decay,
            eps=self.base_config.optimizer_epsilon,
        )

    def get_scheduler(
        self, optimizer: Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
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
