import os
import sys
import logging

from transformers import RobertaTokenizerFast
from model.building import ModelBuilder
from model.trainer import pretrain_and_evaluate
import utils

if __name__ == "__main__":

    training_args, dataset_args, model_args, parser_args = utils.get_args()

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    else: # safeguard to avoid data loss but training_args.output_dir path is generated
        print("Aborting, output dir "+training_args.output_dir+" already exists")
        sys.exit(1)

    log_path = "{0}/run.log".format(training_args.output_dir)

    # Fix logger not writting sometimes
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    logging.info("training args:" + str(training_args))
    logging.info("training output:" + str(training_args.output_dir))
    logging.info("dataset  args:" + str(dataset_args))
    logging.info("model args   :" + str(model_args))
    logging.info("eval only: " + str(parser_args.eval_only))
    logging.info("log:" + log_path)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    builder = ModelBuilder(
        # TODO Load directly from a checkpoint folder without specifying a model cfg
        # to restore both model and trainer state
        path_to_config=parser_args.model,  # optional if a checkpoint is provided
        from_checkpoint=None,  # Directory to load a checkpoint (created by the Trainer)
        vocab_size=len(tokenizer)
    )
    model, config = builder.get_model()

    tokenizer.model_max_length = config.sequence_len
    tokenizer.init_kwargs['model_max_length'] = config.sequence_len

    pretrain_and_evaluate(training_args, dataset_args, model, tokenizer, parser_args.eval_only)
