from typing import Any, Literal, cast
from git import Optional
from pydantic import Field
from src.model.w2vphoneme_head import W2VPhonemeHead, W2VPhonemeHeadArgs
from src.args.base_args import PRETRAINED_LATENT_SIZES
from src.datasets.batch_types import PhonemeSampleBatch
from src.model.b2tmodel import B2TModel, ModelOutput
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
    B2P2TArgsModel,
    B2P2TBrainFeatureExtractorArgsModel,
    W2VBrainEncoderModelArgs,
    W2VPhonemeHeadArgs,
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

    w2v_skip_loading_weights: bool = Field(
        default=False,
        description="Skip loading weights from wav2vec checkpoint, only load architecture",
    )
    intermediate_head_fc_hidden_sizes: list[int] = []
    intermediate_head_fc_activation_function: ACTIVATION_FUNCTION = "gelu"
    intermediate_loss_weight: float = 0.0
    loss_function: Literal["ctc", "combined_ctc"] = "ctc"
    intermediate_loss_squared: Optional[bool] = None
    head_checkpoint: Optional[str] = None


class IntermediateHead(B2TModel):
    def __init__(
        self,
        in_size: int,
        hidden_sizes: list[int],
        hidden_activation: ACTIVATION_FUNCTION,
    ):
        super().__init__()
        self.head = create_fully_connected(
            in_size,
            len(PHONE_DEF_SIL) + 1,
            activation=hidden_activation,
            hidden_sizes=hidden_sizes,
        )
        self.loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def forward(self, batch: PhonemeSampleBatch) -> ModelOutput:
        out = self.head.forward(batch.input)
        targets = batch.target
        assert targets is not None

        targets = torch.where(targets < 1, torch.tensor(-100), targets)

        ctc_loss = (
            self.loss.forward(
                torch.log_softmax(out, -1).transpose(0, 1),
                targets,
                batch.input_lens.cuda(),
                batch.target_lens.cuda(),
            )
            if batch.target_lens is not None and batch.input_lens is not None
            else None
        )
        metrics = {}
        if ctc_loss is not None:
            metrics["ctc_loss"] = ctc_loss.item()
        return ModelOutput(logits=out, metrics=metrics, loss=ctc_loss)


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

        if self.config.intermediate_loss_weight > 0.0:
            assert (
                self.config.loss_function == "combined_ctc"
            ), "When intermediate loss weight is set to larger 0, loss function must be combined_ctc"
        intermediate_head = (
            IntermediateHead(
                PRETRAINED_LATENT_SIZES[self.config.wav2vec_checkpoint],
                self.config.intermediate_head_fc_hidden_sizes,
                self.config.intermediate_head_fc_activation_function,
            )
            if self.config.intermediate_loss_weight > 0.0
            else None
        )
        n_arpabet_phonemes = len(PHONE_DEF_SIL) + 1
        head = W2VPhonemeHead(self.config, n_arpabet_phonemes)
        if self.config.head_checkpoint is not None:
            head.load_state_dict(torch.load(self.config.head_checkpoint))
        model = W2VBrainEncoderModel(
            self.config,
            brain_encoder,
            self.config.wav2vec_checkpoint,
            head=head,
            skip_loading_weights=self.config.w2v_skip_loading_weights,
            pre_w2v_head_for_additional_loss=intermediate_head,
            additonal_loss_weight=self.config.intermediate_loss_weight,
            additional_loss_squared=self.config.intermediate_loss_squared,
        )
        return model

    def create_optimizer(self) -> Optimizer:
        model = cast(W2VBrainEncoderModel, self.model)

        assert model.head is not None, "Head must be defined for this experiment"
        import itertools

        non_w2v_params = (
            model.brain_encoder.parameters()
            if model.pre_w2v_head_for_additional_loss is None
            else itertools.chain(
                model.brain_encoder.parameters(),
                model.pre_w2v_head_for_additional_loss.parameters(),
            )
        )

        trainable_params = [
            {"params": non_w2v_params},
            {
                "params": itertools.chain(
                    model.w2v_encoder.parameters(), model.head.parameters()
                ),
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
