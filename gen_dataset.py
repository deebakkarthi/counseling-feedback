#!/usr/bin/env python3

import os

from datasets import load_dataset

OUTPUT_DIR = "./data/"


def convert_to_text(sample):
    return {"text": f"<s>{sample["text"]}</s>"}


def convert_to_completion(sample):
    ind = sample["text"].find("### Response:")
    completion = sample["text"][ind + 13 :]
    prompt = sample["text"][:ind]
    return {"prompt": prompt, "completion": completion}


if __name__ == "__main__":
    train = load_dataset("avylor/feedback_qesconv", split="train")
    test = load_dataset("avylor/feedback_qesconv", split="test")
    train = train.map(
        convert_to_completion, remove_columns=train.features, batched=False
    )
    dataset = train.train_test_split(test_size=0.1)
    train = dataset["train"]
    valid = dataset["test"]
    print(len(train))
    print(len(test))
    print(len(valid))
    dataset["train"].to_json(os.path.join(OUTPUT_DIR, "train.jsonl"), orient="records")
    dataset["test"].to_json(os.path.join(OUTPUT_DIR, "valid.jsonl"), orient="records")
    test.to_json("./data/" + "test.jsonl", orient="records")
