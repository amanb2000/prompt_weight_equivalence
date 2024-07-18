# Goal. Use huggingface rajpurkar/squad dataset and convert it to a .jsonl file


import json
import os
from tqdm import tqdm
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("squad")

# Save the dataset to a .jsonl file
os.makedirs("data", exist_ok=True)
with open("data/squad_train.jsonl", "w") as f:
    for example in tqdm(dataset["train"]):
        json.dump(example, f)
        f.write("\n")

with open("data/squad_validation.jsonl", "w") as f:
    for example in tqdm(dataset["validation"]):
        json.dump(example, f)
        f.write("\n")