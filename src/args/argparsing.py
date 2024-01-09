import argparse
from src.experiments.b2t_custom_encoder_w2v_experiment import (
    B2TWav2VecCustomEncoderExperiment,
)
from src.experiments.b2t_wav2vec_cnn_experiment import B2TWav2VecCnnExperiment
from src.experiments.b2t_wav2vec_shared_agg_experiment import (
    B2TWav2VecSharedAggregationExperiment,
)
from src.experiments.audio_wav2vec_experiment import AudioWav2VecExperiment
from src.experiments.b2t_audio_wav2vec_experiment import B2TAudioWav2VecExperiment
from src.experiments.b2t_pretraining_wav2vec_experiment import (
    B2TWav2VecPretrainingExperiment,
)
from src.experiments.experiment import Experiment
from pydantic import BaseModel
from src.args.base_args import BaseExperimentArgsModel
from typing import Any, Literal, Type, cast
from src.args.yaml_config import YamlConfig

experiments: dict[str, Type[Experiment]] = {
    "b2t_wav2vec_cnn": B2TWav2VecCnnExperiment,
    "b2t_wav2vec_sharedaggregation": B2TWav2VecSharedAggregationExperiment,
    "audio_wav2vec2": AudioWav2VecExperiment,
    "b2t_audio_wav2vec": B2TAudioWav2VecExperiment,
    "b2t_wav2vec_pretraining": B2TWav2VecPretrainingExperiment,
    "b2t_wav2vec_custom_encoder": B2TWav2VecCustomEncoderExperiment,
}


def str_to_bool(value):
    if value.lower() in ["true", "t"]:
        return True
    elif value.lower() in ["false", "f"]:
        return False
    elif value.lower() in ["none", "n"]:
        return None
    else:
        raise argparse.ArgumentTypeError("Invalid boolean value: {}".format(value))


def _parser_from_model(parser: argparse.ArgumentParser, model: Type[BaseModel]):
    "Add Pydantic model to an ArgumentParser"
    fields = model.__fields__

    for name, field in fields.items():

        def get_type_args():
            is_literal = getattr(field.annotation, "__origin__", None) is Literal
            is_bool = getattr(field.type_, "__name__", None) == "bool"
            if is_literal:
                return {"type": str, "choices": cast(Any, field.annotation).__args__}
            if is_bool:
                return {"type": str_to_bool}
            return {"type": field.type_}

        parser.add_argument(
            f"--{name}",
            dest=name,
            default=field.default,
            help=field.field_info.description,
            **get_type_args(),
        )
    return parser


def _create_arg_parser():
    base_parser = argparse.ArgumentParser()
    base_parser = _parser_from_model(base_parser, BaseExperimentArgsModel)
    base_args, _ = base_parser.parse_known_args()

    experiment_model = experiments[base_args.experiment_type].get_args_model()
    parser = argparse.ArgumentParser(
        description="Machine Learning Experiment Configuration"
    )
    parser = _parser_from_model(parser, experiment_model)
    return parser


def get_experiment_from_args() -> Experiment:
    arg_parser = _create_arg_parser()
    args = arg_parser.parse_args()
    yaml_config = YamlConfig()

    experiment = experiments[args.experiment_type](vars(args), yaml_config.config)
    return experiment
