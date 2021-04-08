from transformers import Trainer
import math
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TextDataset, DataCollatorForLanguageModeling


class Trainer_(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log(self, logs):
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        else:
            logs["epoch"] = 0.

        if "eval_loss" in logs:
            logs["eval_bpc"] = logs["eval_loss"] / math.log(2)
        elif "loss" in logs:
            logs["bpc"] = logs["loss"] / math.log(2)

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        output = {**logs, **{"step": self.state.global_step}}

        # Log everything here
        if output["step"] % self.args.logging_steps == 0:
            logging.info(output)

        self.state.log_history.append(output)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.squeeze(-1)

    labels = np.reshape(labels, (1, -1)).squeeze()
    preds = np.reshape(preds, (1, -1)).squeeze()
    idx = np.where(labels == -100)[0]

    mask = np.ones((len(labels),), dtype=bool)
    mask[idx] = False
    labels = labels[mask, ...]
    preds = preds[mask, ...]

    # TODO discuss average 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)

    metrics = {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    # logging.info(metrics)

    return metrics


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

    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_args.val_datapath,
        block_size=tokenizer.model_max_length,
    )
    if eval_only:
        train_dataset = val_dataset
    else:
        logging.info(
            f"Loading and tokenizing training data is usually slow: {dataset_args.train_datapath}"
        )
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=dataset_args.train_datapath,
            block_size=tokenizer.model_max_length,
        )

    # https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # https://huggingface.co/transformers/_modules/transformers/trainer.html
    trainer = Trainer_(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    # eval_loss = metrics["eval_loss"]
    # logging.info(f"Initial eval bpc: {eval_loss / math.log(2)}")
    logging.info(f"Initial metrics: {metrics}")

    if not eval_only:
        # to change if we want to continue training existing models
        # same path as from_checkpoint argument from the builder
        trainer.train(model_path=None)

        trainer.save_model()

        metrics = trainer.evaluate()
        eval_loss = metrics["eval_loss"]
        logging.info(f"Eval bpc after pretraining: {eval_loss / math.log(2)}")
