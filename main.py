import os, sys
import argparse
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers import RobertaTokenizerFast

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    labels = np.reshape(labels, (1, -1)).squeeze()
    preds = np.reshape(preds, (1, -1)).squeeze()
    idx = np.where(labels == -100)[0]

    mask = np.ones((len(labels),), dtype=bool)
    mask[idx] = False
    labels = labels[mask, ...]
    preds = preds[mask, ...]

    # TODO discuss average 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    metrics = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    logging.info(metrics)

    return metrics

@dataclass
class DatasetArgs:
    val_datapath: str = field(metadata={"help": "validation set"})
    train_datapath: str = field(metadata={"help": "training set"})

def pretrain_and_evaluate(training_args, dataset_args, model, tokenizer, eval_only):
    """
    # adapted from https://colab.research.google.com/drive/1-JIJlao4dI-Ilww_NnTc0rxtp-ymgDgM?usp=sharing#scrollTo=N8J-TLhBuaOf
    :param training_args: HF training args object
    :param dataset_args: object storing dataset config, requires train_datapath and val_datapath to be defined
    :param model: transformers.PreTrainedModel
    :param tokenizer: PreTrainedTokenizerBase
    :param eval_only: boolean, True only performs evaluation
    :return:
    """

    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=dataset_args.val_datapath,
                              block_size=tokenizer.max_len)
    if eval_only:
        train_dataset = val_dataset
    else:
        logging.info(f'Loading and tokenizing training data is usually slow: {dataset_args.train_datapath}')
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=dataset_args.train_datapath,
                                    block_size=tokenizer.max_len)

    # https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # https://huggingface.co/transformers/_modules/transformers/trainer.html
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics)

    metrics = trainer.evaluate()
    eval_loss = metrics["eval_loss"]
    logging.info(f'Initial eval bpc: {eval_loss / math.log(2)}')

    if not eval_only:
        trainer.train(model_path=None)  # to change if we want to continue training existing models
        trainer.save_model()

        metrics = trainer.evaluate()
        eval_loss = metrics["eval_loss"]
        logging.info(f'Eval bpc after pretraining: {eval_loss / math.log(2)}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='XP FTF')
    parser.add_argument('--model', help='path model configuration', type=str)
    parser.add_argument('--train', help='path trainer configuration', type=str)
    parser.add_argument('--dataset', help='dataset configuration', type=str)
    parser.add_argument('--eval_only', help='only perform evaluation', default=False, action='store_true')

    # TODO delete before experiments
    # args = parser.parse_args() # load parameters from sys.argv
    args = parser.parse_args(
        ['--model', 'model_file', '--train', 'config/training_config.txt', '--dataset', 'config/dataset_config.txt'])

    # https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py
    parser_hf_trainer_args = HfArgumentParser((TrainingArguments,))
    training_args = parser_hf_trainer_args.parse_args_into_dataclasses(args_filename=args.train)[0]

    parser_hf_dataset_args = HfArgumentParser((DatasetArgs,))
    dataset_args = parser_hf_dataset_args.parse_args_into_dataclasses(args_filename=args.dataset)[0]

    # TODO Load json file dedicated to the model
    # modification can be done to use a @dataclass as done for dataset_args
    model_args = {"name": "roberta"}

    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    # TODO add model info in log file name
    log_path = "{0}/{1}.log".format(training_args.output_dir, model_args["name"])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    # TODO load model using preloaded configuration
    # model = load_model(args.model)
    # args.model must be a dictionary with keys name/
    # TODO delete after substitution
    from transformers import RobertaForMaskedLM
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    # END DELETE

    eval_only = args.eval_only

    logging.info("training args:" + str(training_args))
    logging.info("dataset  args:" + str(dataset_args))
    logging.info("model args   :" + str(model_args))
    logging.info("eval only: " + str(eval_only))
    logging.info(model)

    pretrain_and_evaluate(training_args, dataset_args, model, tokenizer, eval_only)
