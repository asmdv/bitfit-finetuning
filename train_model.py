"""
Code for Problem 1 of HW 2.
"""
import pickle
from typing import Any, Dict

import evaluate
import numpy as np
import optuna
from datasets import Dataset, load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction


def batch_tokenizer(batch, tokenizer):
    output = tokenizer(batch["text"], padding='max_length', truncation=True, max_length=512)
    batch.update(output)
    return batch


def get_tokenizer(model_name):
    return BertTokenizerFast.from_pretrained(model_name)


def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) \
        -> Dataset:
    return dataset.map(lambda batch: batch_tokenizer(batch, tokenizer), batched=True)


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=2)
    if use_bitfit:
        for name, param in model.named_parameters():
            if not ('bias' in name or 'classifier' in name):
                param.requires_grad = False
    return model


def compute_objective(metrics):
    return metrics['eval_accuracy']


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
    def model_init(trial):
        return init_model(trial, model_name, use_bitfit)

    tokenizer = get_tokenizer(model_name)
    args = TrainingArguments(output_dir="checkpoints",
                             num_train_epochs=4,
                             per_device_eval_batch_size=128,
                             evaluation_strategy='epoch',
                             save_strategy='epoch',
                             save_steps=1,
                             save_total_limit=3,
                             load_best_model_at_end=True
                             )

    return Trainer(
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        model_init=model_init,
        compute_metrics=compute_metrics
    )


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [3e-4, 1e-4, 5e-5, 3e-5]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size",
                                                                 [8, 16, 32, 64, 128]),
    }


def search_space():
    return {
        "learning_rate": [3e-4, 1e-4, 5e-5, 3e-5],
        "per_device_train_batch_size": [8, 16, 32, 64, 128]
    }


def hyperparameter_search_settings() -> Dict[str, Any]:
    return dict(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        sampler=optuna.samplers.GridSampler(search_space()),
        compute_objective=compute_objective
    )


if __name__ == "__main__":  # Use this script to train your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = get_tokenizer(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    # Set up trainer
    trainer = init_trainer(model_name, imdb["train"], imdb["val"],
                           use_bitfit=True)

    # Train and save the best hyperparameters
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open("train_results.p", "wb") as f:
        pickle.dump(best, f)
