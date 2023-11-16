import argparse
from src.experiments.wav2vec import Wav2VecExperiment
from src.experiments.experiment import Experiment
from pydantic import BaseModel
from src.args.base_args import BaseArgsModel

experiments: dict[str, Experiment] = {"wav2vec": Wav2VecExperiment}


def _parser_from_model(parser: argparse.ArgumentParser, model: BaseModel):
    "Add Pydantic model to an ArgumentParser"
    fields = model.__fields__
    for name, field in fields.items():
        parser.add_argument(
            f"--{name}",
            dest=name,
            type=field.type_,
            default=field.default,
            help=field.field_info.description,
        )
    return parser


def _create_arg_parser():
    base_parser = argparse.ArgumentParser()
    base_parser = _parser_from_model(base_parser, BaseArgsModel)
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

    experiment = experiments[args.experiment_type](vars(args))
    return experiment
