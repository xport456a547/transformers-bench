import argparse
from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser
import json
import sys, os
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser(description="Fast Transformers experiments")
    parser.add_argument("--model", help="model configuration file path (default: config/model.json)", type=str)
    parser.add_argument("--train", help="trainer configuration file path (default: config/training_config.txt)",
                        type=str)
    parser.add_argument("--dataset", help="dataset configuration  file path (default: config/dataset_config.txt)",
                        type=str)
    parser.add_argument(
        "--eval_only",
        help="only perform evaluation",
        default=False,
        action="store_true",
    )
    return parser


@dataclass
class DatasetArgs:
    name: str = field(metadata={"help": "dataset name (used to name result dir)"})
    val_datapath: str = field(metadata={"help": "validation set"})
    train_datapath: str = field(metadata={"help": "training set"})


def get_args():

    parser = get_parser()

    # load arguments from sys.argv or default location (cf. parser doc)
    if len(sys.argv) != 1:
        args = parser.parse_args()
    else:
        args = parser.parse_args(
            [
                "--model",
                "config/model.json",
                "--train",
                "config/training_config.txt",
                "--dataset",
                "config/dataset_config.txt",
            ]
        )

    # https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py
    parser_hf_trainer_args = HfArgumentParser((TrainingArguments,))
    training_args = parser_hf_trainer_args.parse_args_into_dataclasses(
        args_filename=args.train
    )[0]

    parser_hf_dataset_args = HfArgumentParser((DatasetArgs,))
    dataset_args = parser_hf_dataset_args.parse_args_into_dataclasses(
        args_filename=args.dataset
    )[0]

    assert os.path.isfile(args.model)
    model_args = json.load(open(args.model, "r"))

    # redefine training output dir to include model information and avoid data loss

    output_dir_base = "{0}/{1}_{2}".format(training_args.output_dir,
                                                    dataset_args.name, model_args["model"])
    training_args.output_dir = output_dir_base

    while os.path.isdir(training_args.output_dir):
        date_time = datetime.now().strftime("%m%d%Y_%H%M%S_%f")
        training_args.output_dir = "{0}_{1}".format(output_dir_base, date_time)
        import time
        time.sleep(1)

    return training_args, dataset_args, model_args, args
