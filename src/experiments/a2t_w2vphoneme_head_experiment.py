from typing import Any, Literal, cast

from src.datasets.timit_a2t_seq_dataset import TimitA2TSeqDataset
from src.datasets.batch_types import SampleBatch
from src.experiments.experiment import Experiment
from src.model.b2tmodel import ModelOutput
from src.experiments.b2t_gru_w2v_conformer_experiment import W2V_CHECKPOINT_TO_PROCESSOR
from src.args.yaml_config import YamlConfigModel
from torch.optim.optimizer import Optimizer
from src.datasets.audio_with_phonemes_seq import (
    AudioWPhonemesDatasetArgsModel,
)
from src.train.evaluator import EvaluatorWithW2vLMDecoder
from src.args.base_args import BaseExperimentArgsModel
from src.model.w2vphoneme_head import (
    W2VPhonemeHead,
    W2VPhonemeHeadArgs,
    W2VPhonemeWithCustomHead,
)
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from src.train.history import DecodedPredictionBatch

# Baseline Experiment: b2p2t_gru:
# 5gram + rescoring: 0.28 WER (/hpi/fs00/scratch/tobias.fiedler/brain2text/experiment_results/b2p2t_gru/2024-03-27_17#31#07),
# 3gram: 0.3153 WER
# (/hpi/fs00/scratch/tobias.fiedler/brain2text/experiment_results/b2p2t_gru/2024-03-28_08#18#39)

W2V_CHECKPOINT_TO_PROCESSOR = {
    "facebook/wav2vec2-lv-60-espeak-cv-ft": "patrickvonplaten/wav2vec2-base-100h-with-lm",
}


class A2T_W2VPhonemeHeadExperimentArgs(
    BaseExperimentArgsModel, AudioWPhonemesDatasetArgsModel, W2VPhonemeHeadArgs
):
    # See https://huggingface.co/models?other=wav2vec2 for available checkpoints
    wav2vec_checkpoint: Literal["facebook/wav2vec2-lv-60-espeak-cv-ft"] = (
        "facebook/wav2vec2-lv-60-espeak-cv-ft"
    )
    unfreeze_strategy: Literal["head"] = "head"
    tokenizer_checkpoint = "facebook/wav2vec2-base-960h"  # Differs from wav2vec_checkpoint since this is for the actual labels which we want to map the given wav2vec checkpoint to
    lm_decode_test_predictions: bool = False


class A2T_W2VPhonemeHeadExperiment(Experiment):
    def __init__(self, config: dict, yamlConfig: YamlConfigModel):
        self.config = self.get_args_model()(**config)
        self.yaml_config = yamlConfig
        self.tokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(
                self.config.tokenizer_checkpoint,
                cache_dir=self.yaml_config.cache_dir,
                use_fast=self.config.use_fast_tokenizer,
            ),
        )
        super().__init__(config, yamlConfig)

    def get_name(self) -> str:
        return "a2t_w2vphoneme_head"

    @staticmethod
    def get_args_model():
        return A2T_W2VPhonemeHeadExperimentArgs

    def _create_model(self):
        assert (
            self.config.loss_function == "ctc",  # type: ignore
            "Only ctc loss is supported",
        )
        head = W2VPhonemeHead(self.config, out_size=self.tokenizer.vocab_size)
        model = W2VPhonemeWithCustomHead(self.config.wav2vec_checkpoint, head)
        return model

    def get_vocab(self) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(
            list(range(self.tokenizer.vocab_size))
        )

    def _create_dataset(self, split: Literal["train", "val", "test"] = "train"):
        return TimitA2TSeqDataset(
            self.config, self.yaml_config, split=split, tokenizer=self.tokenizer
        )

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            if self.config.unfreeze_strategy == "head":
                return cast(W2VPhonemeWithCustomHead, self.model).head.parameters()
            raise Exception(
                f"Unfreeze strategy {self.config.unfreeze_strategy} is not implemented for {self.get_name()} experiment"
            )

        optim: Any = self._get_optimizer_cls()
        return optim(get_trainable_params(), lr=self.config.learning_rate)

    def decode_predictions(
        self, predictions: ModelOutput, sample: SampleBatch
    ) -> DecodedPredictionBatch:
        predicted_ids = predictions.logits.argmax(dim=-1).cpu().numpy()
        predicted_strings = self.tokenizer.batch_decode(
            predicted_ids, group_tokens=True
        )
        label_strings = (
            self.tokenizer.batch_decode(sample.target.cpu().numpy(), group_tokens=False)
            if sample.target is not None
            else None
        )
        return DecodedPredictionBatch(predicted_strings, label_strings)

    def create_evaluator(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ):
        return EvaluatorWithW2vLMDecoder(
            self.tokenizer,
            mode,
            self.yaml_config.cache_dir,
            W2V_CHECKPOINT_TO_PROCESSOR[self.config.wav2vec_checkpoint],
            track_non_test_predictions,
            self.config.lm_decode_test_predictions,
        )
