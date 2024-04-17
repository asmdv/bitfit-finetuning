"""
Code for Problem 1 of HW 2.
"""
import pickle

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments
import numpy as np
from train_model import preprocess_dataset


def get_tokenizer(model_name):
    return BertTokenizerFast.from_pretrained(model_name)


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def init_tester(directory: str) -> Trainer:
    model = BertForSequenceClassification.from_pretrained(directory)
    bitfit_sum = 0
    total_sum = 0
    
    for name, param in model.named_parameters():
        if 'bias' in name or 'classifier' in name:
            bitfit_sum += param.numel()
        total_sum += param.numel()
        
    print("Number of parameters (Bitfit=True):", bitfit_sum)
    print("Number of parameters (Bitfit=False):", total_sum)

    args = TrainingArguments(
        output_dir="tests",
        do_predict=True,
        per_device_eval_batch_size=128
    )

    return Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics
    )


if __name__ == "__main__":
    model_name = "prajjwal1/bert-tiny"

    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    tokenizer = get_tokenizer(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    tester = init_tester("checkpoints_bitfit_false/run-0/checkpoint-1252")
    

    results = tester.predict(imdb["test"])
    print("Test accuracy: ", results.metrics['test_accuracy'])
    with open("test_results.p", "wb") as f:
        pickle.dump(results, f)
