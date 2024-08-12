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

from src.train.evaluator import DefaultEvaluator
from src.datasets.batch_types import SampleBatch
from src.model.b2tmodel import ModelOutput
from src.train.history import DecodedPredictionBatch, MetricEntry, SingleEpochHistory
from torcheval.metrics import WordErrorRate
from transformers import PreTrainedTokenizer
from transformers import Wav2Vec2ProcessorWithLM

# Baseline Experiment: b2p2t_gru:
# 5gram + rescoring: 0.28 WER (/hpi/fs00/scratch/tobias.fiedler/brain2text/experiment_results/b2p2t_gru/2024-03-27_17#31#07),
# 3gram: 0.3153 WER
# (/hpi/fs00/scratch/tobias.fiedler/brain2text/experiment_results/b2p2t_gru/2024-03-28_08#18#39)


class EnhancedDecodedBatch(DecodedPredictionBatch):
    predictions_lm_decoded: list[str]


W2V_CHECKPOINT_TO_PROCESSOR = {
    "facebook/wav2vec2-base-960h": "patrickvonplaten/wav2vec2-base-100h-with-lm",
    "facebook/wav2vec2-base-100h": "patrickvonplaten/wav2vec2-base-100h-with-lm",
    "jonatasgrosman/wav2vec2-large-xlsr-53-english": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "facebook/wav2vec2-large-960h": "patrickvonplaten/wav2vec2-base-100h-with-lm",  # we can (probably) use the same processor as for the 100h model, as the outputs of W2V are the same
}


class B2TGruAndW2VExperimentEvaluator(DefaultEvaluator):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mode: Literal["train", "val", "test"],
        cache_dir: str,
        w2v_checkpoint: str,
        track_non_test_predictions: bool = False,
        lm_decode_test_predictions: bool = False,
    ):
        super().__init__(tokenizer, mode, track_non_test_predictions)
        self.history = SingleEpochHistory()
        self.tokenizer = tokenizer

        # Tutorial: https://huggingface.co/blog/wav2vec2-with-ngram
        processor_checkpoint = W2V_CHECKPOINT_TO_PROCESSOR[w2v_checkpoint]
        assert (
            processor_checkpoint is not None
        ), f"Processor for {w2v_checkpoint} not found"
        self.processor = (
            Wav2Vec2ProcessorWithLM.from_pretrained(
                processor_checkpoint, cache_dir=cache_dir
            )
            if lm_decode_test_predictions and mode == "test"
            else None
        )

    def _track_batch(self, predictions: ModelOutput, sample: SampleBatch):
        predicted_strings, label_strings = self.decode_predictions(predictions, sample)

        # remove characters after EOS token
        def cut_after_eos_token(string: str):
            eos_token = "</s>"
            index_of_eos = string.find(eos_token)
            if index_of_eos != -1:
                return string[: (index_of_eos + len(eos_token))]
            else:
                return string

        predicted_strings = [
            cut_after_eos_token(string) for string in predicted_strings
        ]
        decoded_batch = EnhancedDecodedBatch(
            predictions=predicted_strings, targets=label_strings
        )

        if label_strings is not None:
            additional_metrics = {
                "word_error_rate": WordErrorRate()
                .update(input=predicted_strings, target=label_strings)
                .compute()
                .item()
            }
            if self.processor is not None and self.mode == "test":
                processed = self.processor.batch_decode(
                    predictions.logits.detach().cpu().numpy()
                )
                additional_metrics["word_error_rate_lm_decode"] = (
                    WordErrorRate()
                    .update(input=cast(list[str], processed.text), target=label_strings)
                    .compute()
                    .item()
                )
                decoded_batch.predictions_lm_decoded = cast(list[str], processed.text)

            predictions.metrics.update(additional_metrics)

        assert (
            predictions.loss != None
        ), "Loss is None. Make sure to set loss in ModelOutput"

        self.history.add_batch_metric(
            MetricEntry(predictions.metrics, predictions.loss.cpu().item()),
            (
                decoded_batch
                if self.mode == "test" or self.track_non_test_predictions
                else None
            ),
        )


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
    wav2vec_checkpoint: str = (
        "facebook/wav2vec2-base-960h"  # "jonatasgrosman/wav2vec2-large-xlsr-53-english" OR "facebook/wav2vec2-large-960h"
    )
    lm_decode_test_predictions: bool = False


class B2TGruAndW2VExperiment(B2TExperiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        super().__init__(config, yamlConfig)

        if self.config.tokenizer_checkpoint != self.config.wav2vec_checkpoint:
            print(
                f"Tokenizer checkpoint ({self.config.tokenizer_checkpoint}) is different to wav2vec_checkpoint ({self.config.wav2vec_checkpoint}). This may lead to unexpected behaviour"
            )

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

    def create_evaluator(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        return B2TGruAndW2VExperimentEvaluator(
            self.tokenizer,
            mode,
            self.yaml_config.cache_dir,
            self.config.wav2vec_checkpoint,
            track_non_test_predictions,
            self.config.lm_decode_test_predictions,
        )
