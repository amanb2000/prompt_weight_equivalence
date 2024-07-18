# Goal: init model, run lora to finetune weights, save model

import os
import json
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from datetime import datetime
import argparse
import pdb


# Llama 3 system prompt + user prompt
# num_epochs = 1000
# batch_size = 10
# learning_rate = 1e-4
# device = "cuda" if torch.cuda.is_available() else "cpu"
# data_path = "data/traj_lex_nseq1000_maxlen300_minlen100_temp2.0.jsonl"
# out_dir = "results/traj_lex_01"

# Set up argument parser
parser = argparse.ArgumentParser(description="Script for training model with specified parameters")

# Add arguments with defaults
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train the model. Default = 1000')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training. Default = 10')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer. Default = 1e-4')
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to run the training on. Default = "cuda" if available, else "cpu"')
parser.add_argument('--data_path', type=str, default="data/traj_lex_nseq1000_maxlen300_minlen100_temp2.0.jsonl", help='Path to the training data. Default = "data/traj_lex_nseq1000_maxlen300_minlen100_temp2.0.jsonl"')
parser.add_argument('--out_dir', type=str, default="results/traj_lex_01", help='Output directory for results. Default = "results/traj_lex_01"')

# Parse arguments
args = parser.parse_args()

# Assign variables
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
device = args.device
data_path = args.data_path
out_dir = args.out_dir

# Ensure the output directory exists
os.makedirs(out_dir, exist_ok=True)

# Save arguments to a JSON file in the output directory
args_dict = vars(args)
args_json_path = os.path.join(out_dir, 'args.json')
with open(args_json_path, 'w') as f:
    json.dump(args_dict, f, indent=4)

print(f"Arguments saved to {args_json_path}")






# make the out_dir if it doesn't already exist 
import os
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
def log(msg, file_path): 
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a') as f: 
        f.write(f"[{current_time}] {msg}\n")

log_path = os.path.join(out_dir, "train_loop.log")

# TODO Load dataset here
# dataset = []
# with open(data_path, "r") as f:
#     for line in f:
#         newline_str = line.strip()
#         newline_dict = eval(newline_str)
#         dataset.append(newline_dict)

# Load the dataset from the JSONL file
log("Loading dataset", log_path)
dataset = load_dataset('json', data_files=data_path)
log("Dataset loaded", log_path)




if __name__ == "__main__":
    # Load model
    
    # Initialize a tokenizer and model
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    log(f"Loading model {model_name}...", log_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = pipeline.model

    log("Model loaded", log_path)

    log("Loading PEFT model...", log_path)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=1,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", 
            "v_proj", 
            "k_proj", 
            "o_proj"
        ]
    )
    
    peft_model = get_peft_model(model, peft_config)
    print("unprompted model parameter stats:")
    peft_model.print_trainable_parameters()
    log("PEFT model loaded", log_path)


    pipeline_prompted_llm = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    prompted_llm = pipeline_prompted_llm.model
    prompted_llm.eval()

    device = model.device

    optimizer = torch.optim.Adam(peft_model.parameters(), lr=learning_rate)

    # Run LORA
    # We are going to run both the unprompted model and the prompted model on the same batch of inputs
    # Then, we are going to compute the KL divergence between the two models' logits *for all tokens which BOTH models see*.
    # Therefore, we are not going to compute the KL divergence for the prompt that only the prompted model sees.
    # We are going to use the KL divergence as the loss function for LORA.
    # We are going to backpropagate through the unprompted model and update its weights.
    # The goal is to force the unprompted model to generate the same logits as the prompted model for the tokens that both models see.


    for epoch in range(num_epochs):
        log("Started epoch " + str(epoch), log_path)
        for i in range(0, len(dataset), batch_size):
            log(f"Batch {i}", log_path)
            batch = dataset['train'][i:i+batch_size]
            input_ids = batch['input_ids']

            main_text_ids = [[128000] + x[28:] for x in input_ids]

            inputs_unprompted = torch.tensor(main_text_ids).to(device)
            print("Unprompted inputs shape", inputs_unprompted.shape)
            # Concatenate the prompt to the input for the prompted model for all inputs in the batch
            inputs_prompted = torch.tensor(input_ids).to(device)
            print("Prompted inputs shape", inputs_prompted.shape)

            log("Computing unprompted logits...", log_path)
            unprompted_logits = peft_model(inputs_unprompted).logits
            log("Done computing prompted logits...", log_path)

            log("Computing prompted logits...", log_path)
            with torch.no_grad(): 
                prompted_logits = prompted_llm(inputs_prompted).logits[:, -inputs_unprompted.shape[1]:, :]
            log("Done computing prompted logits...", log_path)

            # Compute KL divergence: KL(prompted, unprompted)
            kl_div = F.kl_div(F.log_softmax(unprompted_logits, dim=-1), 
                            F.log_softmax(prompted_logits, dim=-1), 
                            reduction='batchmean', 
                            log_target=True)
            
            kl_div.backward()
            optimizer.step()
            optimizer.zero_grad()
            log(f"Done computing KL divergence = {kl_div.item()}", log_path)

        print(f"Epoch {epoch} loss: {kl_div.item()}")
        log(f"Epoch {epoch} loss: {kl_div.item()}", log_path)

    # save model
    log(f"Saving PEFT model to {out_dir}...", log_path)
    peft_model.save_pretrained(out_dir)
    log(f"Model saved to {out_dir}", log_path)
