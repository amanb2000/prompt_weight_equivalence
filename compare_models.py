"""
Given the location of a trained model (the --out_dir arg for train_loop.py) and 
a .jsonl dataset generated by generate_data.py, this script will compare the 
-log likelihood of trajectories in the dataset under the trained model and the 
prompted model. 

Specifically, we will compute the -log likelihood of all trajectories for the 
following 4 combinations of {model, dataset}: 
 1. PEFT model, no prompt (how well can fine-tuned model generalize)? 
 2. Base model, with prompt (original prompted model).
 3. PEFT model, with prompt (who's to say what will happen here -- kinda like 2x-ing the prompt's influence). 
 4. Base model, no prompt (pre-base line)

Then we will draw scatter plots for all combinations of (1-4) x (1-4) and save 
to the `--results_dir` (i.e., the same dir as given in the --out_dir in train_loop.py).
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

import sys 
import os
from datetime import datetime
import json

from tqdm import tqdm

from generate_data import format_prompt
from train_loop import pad_list_of_lists

import plotly.express as px
import pandas as pd
import numpy as np

import pdb

def parse_args(): 
    parser = argparse.ArgumentParser(description='Compare trained model to prompted model')
    parser.add_argument('--results_dir', type=str, required=True, help='Location of trained model, as given in --out_dir in train_loop.py (or one of the subdirectories with an epoch_N save point).')
    parser.add_argument('--data_file', type=str, default="NONE", help='Location of .jsonl dataset. If NONE, we will pull the args.json from results_dir and use the validation dataset specified there.')
    parser.add_argument('--u_override', type=str, default=None, help="Path to a .md file containing the string for the control input u (system prompt) override. This will be 'surgically inserted' between the <|start_header_id|>system<|end_header_id|>\n\n[--u_override GOES HERE]<|eot_id|>\n<|start_header_id|>user<|end_header_id|>. THIS ONLY WORKS FOR LLAMA-3 MODELS AND OTHER MODELS WITH THE SAME TOKENIZER AND PROMPT FORMAT.")
    parser.add_argument('--path_prefix', type=str, default=None, help="Prefix for saved files. Helpful for clarifying if you used --u_override. The rest of the saved figures names will specify only whether it's the PEFT model or the base model and whether it is prompted or unprompted.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for computing log likelihoods. Default=8.")
    args =  parser.parse_args()

    # pretty print args 
    print("Args:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    
    return args

def log(msg): 
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log.log_path, 'a') as f: # log_path is a function attribute of log.
        f.write(f"[{current_time}] {msg}\n")


def get_og_args(results_dir): 
    # read the args.json from the results_dir
    args_path = os.path.join(results_dir, "args.json")
    with open(args_path, 'r') as f: 
        args = json.load(f)
    
    return args

def get_models(results_dir, device='cuda'):
    # get the PEFT model, regular model, and tokenizer

    # start with the untrained_model
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

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
    base_model = pipeline.model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # get the PEFT model
    base_model_ = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    config = PeftConfig.from_pretrained(results_dir)

    # Load the PEFT model
    peft_model = PeftModel.from_pretrained(base_model_, results_dir).to(device)

    print("PEFT model loaded successfully.")

    return base_model, peft_model, tokenizer


def get_u_override(u_override_path, tokenizer): 
    # get the string and the input_ids WITH NO BOS OR EOS TOKENS 
    with open(u_override_path, 'r') as f: 
        u_str = f.read()

    log(f"Loaded u_str='{u_str}' from {u_override_path}")

    # tokenize the u_str
    u_list = tokenizer(u_str)['input_ids'] # this is a 1-d list
    log(f"u_list is: {u_list}")

    # remove the BOS and EOS tokens -- any tokens in tokenizer.special_tokens.
    u_list = [x for x in u_list if x not in (128000, 128001, 128009)]

    log(f"After removing BOS and EOS tokens, u_list is: '{u_list}'")
    log(f"After removing BOS and EOS tokens, tokenizer.decode(u_str) = '{tokenizer.decode(u_list)}'")

    return u_str, u_list


def transplant_u(u_list, input_ids_list, mask_list, tokenizer):
    """ Surgically implants u_list into input_ids_list and mask_list to replace 
    the existing text encased in the <|start_header_id|>system<|end_header_id|> 
    and <|start_header_id|>user<|end_header_id|> tokens.
    """
    assert len(input_ids_list) == len(mask_list), "input_ids_list and mask_list must have the same length"

    # FOR FUTURE REFERENCE -- we used the following tokens to encode the system and user prompts. 
    # follows from the generate_dataset.py output format. 
    # 
    # [1:] gets rid of the BOS token
    # system_tokens = tokenizer.encode("<|start_header_id|>system<|end_header_id|>\n\n")[1:]
    # user_tokens = tokenizer.encode("<|eot_id|>\n<|start_header_id|>user<|end_header_id|>")[1:]

    system_tokens = [128006, 9125, 128007, 271] # use system_tokens[:-1] for lazy people who don't include \n\n
    user_tokens = [128009, 198, 128006, 882, 128007] # use user_tokens[2:] for lazy people who don't include <|eot_id|>\n 


    for i in range(len(input_ids_list)):
        assert len(input_ids_list[i]) == len(mask_list[i]), f"input_ids_list[{i}] and mask_list[{i}] must have the same length"
        # find the index of system_tokens inside input_ids_list[i]
        system_end = None
        for j in range(len(input_ids_list[i])):
            if input_ids_list[i][j:j+len(system_tokens)] == system_tokens:
                system_end = j+len(system_tokens)
                break
        
        assert system_end is not None, "Could not find system_tokens in input_ids_list"

        user_start = None 
        for j in range(system_end, len(input_ids_list[i])):
            if input_ids_list[i][j:j+len(user_tokens)] == user_tokens:
                user_start = j
                break
        
        assert user_start is not None, "Could not find user_tokens in input_ids_list"

        input_ids_list[i] = input_ids_list[i][:system_end] + u_list + input_ids_list[i][user_start:]
        mask_list[i] = mask_list[i][:system_end] + [0]*len(u_list) + mask_list[i][user_start:]

    return input_ids_list, mask_list
    

def get_loss(prompted_llm, peft_model, tokenizer,
             dataset, 
             batch_size, 
             optimizer, 
             do_step = True, 
             u_override = None):
    """
    Gets the loss for {prompted_llm, peft_model} on the {prompted, unprompted} 
    inputs_ids from a dataset. 

    We retain the batch dimension so that we have the loss for every individual 
    entry in the dataset
    """
    print("peft model device: ", peft_model.device)
    print("prompted model device: ", prompted_llm.device)

    if u_override is not None:
        log(f"Using u_override from {u_override}")
        u_str, u_list = get_u_override(u_override, tokenizer)


    unprompted_logits_peft_losses = []
    unprompted_logits_base_losses = []
    prompted_logits_peft_losses = []
    prompted_logits_base_losses = []
    texts = []
    lengths = []

    for i in tqdm(range(0, len(dataset['train']), batch_size)):
        batch = dataset['train'][i:i+batch_size]
        # grab the input_ids_nosys to run thru the PEFT model 
        input_ids_nosys_list_ = batch['input_ids_nosys'] # pad with tokenizer.pad_token_id
        input_ids_list_ = batch['input_ids'] # pad with tokenizer.pad_token_id

        input_ids_nosys_list = pad_list_of_lists(input_ids_nosys_list_, tokenizer.pad_token_id, verbose=False)
        input_ids_list = pad_list_of_lists(input_ids_list_, tokenizer.pad_token_id, verbose=False)

        # grab masks forr each input_ids
        mask_nosys_list_ = batch['generated_text_mask_nosys'] # pad with 0
        mask_list_ = batch['generated_text_mask'] # pad with 0

        mask_nosys_list = pad_list_of_lists(mask_nosys_list_, 0, verbose=False)
        mask_list = pad_list_of_lists(mask_list_, 0, verbose=False)

        # If the user specified --u_override, we will insert the override into 
        # input_ids_list and mask_list. 
        # The override WILL NOT be applied to input_ids_nosys_list and mask_nosys_list. 
        # 
        # We can ensure that the override is applied to the correct indices by 
        # comparing input_ids[mask] with input_ids_nosys[mask_nosys]. 
        # Recall that the mask selects for only the tokens in the trajectory 
        # (i.e., the generated text and not the system prompt u).
        # This is done below once we convert to tensors.

        if u_override is not None:
            input_ids_list, mask_list = transplant_u(u_list, input_ids_list, mask_list, tokenizer)

        device = prompted_llm.device
        input_ids = torch.tensor(input_ids_list).to(device)
        input_ids_nosys = torch.tensor(input_ids_nosys_list).to(device)
        mask = torch.tensor(mask_list).to(device) == 1
        mask_nosys = torch.tensor(mask_nosys_list).to(device) == 1

        assert input_ids.shape == mask.shape
        assert input_ids_nosys.shape == mask_nosys.shape

        assert (input_ids[mask] != input_ids_nosys[mask_nosys]).sum() == 0, "Prompted and unprompted input_ids do not match within their respective masks for the generated text (must be identical)"

        with torch.no_grad(): 
            unprompted_logits_peft = peft_model(input_ids_nosys).logits[:, :-1, :]
            unprompted_logits_base = prompted_llm(input_ids_nosys).logits[:, :-1, :]

            prompted_logits_peft = peft_model(input_ids).logits[:, :-1, :]
            prompted_logits_base = prompted_llm(input_ids).logits[:, :-1, :]

        # now we compute CE loss for each token in the generated text. 
        # for `unprompted_logits_peft`, we use `mask_nosys`  
        # for `unprompted_logits_base`, we use `mask_nosys`
        # for `prompted_logits_peft`, we use `mask`
        # for `prompted_logits_base`, we use `mask`
        # 
        # We must retain the batch dimension, such that we have one 
        # list element in unprompted_logits_peft_losses ... text for every 
        # dataset element.
        # 
        # Calculate losses
        unprompted_peft_loss = F.cross_entropy(unprompted_logits_peft.transpose(1, 2), input_ids_nosys[:, 1:], reduction='none')
        unprompted_base_loss = F.cross_entropy(unprompted_logits_base.transpose(1, 2), input_ids_nosys[:, 1:], reduction='none')
        prompted_peft_loss = F.cross_entropy(prompted_logits_peft.transpose(1, 2), input_ids[:, 1:], reduction='none')
        prompted_base_loss = F.cross_entropy(prompted_logits_base.transpose(1, 2), input_ids[:, 1:], reduction='none')

        # Apply masks and sum losses for each example
        unprompted_peft_loss = (unprompted_peft_loss * mask_nosys[:, 1:]).sum(dim=1) / mask_nosys.sum(dim=1)
        unprompted_base_loss = (unprompted_base_loss * mask_nosys[:, 1:]).sum(dim=1) / mask_nosys.sum(dim=1)
        prompted_peft_loss = (prompted_peft_loss * mask[:, 1:]).sum(dim=1) / mask.sum(dim=1)
        prompted_base_loss = (prompted_base_loss * mask[:, 1:]).sum(dim=1) / mask.sum(dim=1)

        # Append losses to respective lists
        unprompted_logits_peft_losses.extend(unprompted_peft_loss.cpu().tolist())
        unprompted_logits_base_losses.extend(unprompted_base_loss.cpu().tolist())
        prompted_logits_peft_losses.extend(prompted_peft_loss.cpu().tolist())
        prompted_logits_base_losses.extend(prompted_base_loss.cpu().tolist())

        # Collect texts
        texts.extend(batch['text'])

        # Collect lengths
        # lengths are given by the number of 1-valued mask elements
        lengths.extend(mask.sum(dim=1).cpu().tolist())
    
    return {
        'unprompted_logits_peft_losses': unprompted_logits_peft_losses,
        'unprompted_logits_base_losses': unprompted_logits_base_losses,
        'prompted_logits_peft_losses': prompted_logits_peft_losses,
        'prompted_logits_base_losses': prompted_logits_base_losses,
        'lengths': lengths,
        'texts': texts
    }

def compute_deviance(prompted_llm, peft_model, tokenizer, dataset, batch_size, device='cuda'):
    """ 
    Get logits for {prompted_llm, peft_model} on the {prompted, unprompted} 
    inputs_ids from a dataset. 

    We retain the batch dimension so that we have the loss for every individual 
    entry in the dataset
    """

    delta = {}
    delta_prime = {}
    for j in tqdm(range(0, len(dataset['train']), batch_size)):
        batch = dataset['train'][j:j+batch_size]
        # grab the input_ids_nosys to run thru the PEFT model 
        input_ids_nosys_list_ = batch['input_ids_nosys'] # pad with tokenizer.pad_token_id
        input_ids_list_ = batch['input_ids'] # pad with tokenizer.pad_token_id

        input_ids_nosys_list = pad_list_of_lists(input_ids_nosys_list_, tokenizer.pad_token_id, verbose=False)
        input_ids_list = pad_list_of_lists(input_ids_list_, tokenizer.pad_token_id, verbose=False)

        # grab masks forr each input_ids
        mask_nosys_list_ = batch['generated_text_mask_nosys'] # pad with 0
        mask_list_ = batch['generated_text_mask'] # pad with 0

        mask_nosys_list = pad_list_of_lists(mask_nosys_list_, 0, verbose=False)
        mask_list = pad_list_of_lists(mask_list_, 0, verbose=False)
        unprompted_masks = mask_nosys_list
        prompted_masks = mask_list
        unprompted_masks_tensor = torch.tensor(unprompted_masks).to(device)
        prompted_masks_tensor = torch.tensor(prompted_masks).to(device)

        device = prompted_llm.device
        input_ids = torch.tensor(input_ids_list).to(device)
        input_ids_nosys = torch.tensor(input_ids_nosys_list).to(device)

        with torch.no_grad(): 
            unprompted_logits_peft = peft_model(input_ids_nosys).logits
            unprompted_logits_base = prompted_llm(input_ids_nosys).logits

            prompted_logits_peft = peft_model(input_ids).logits
            prompted_logits_base = prompted_llm(input_ids).logits

        max_len_sequence = max(sum(x) for x in mask_list)
        unprompted_start_idx = (torch.arange(0, input_ids.shape[0]).to(prompted_llm.device), unprompted_masks_tensor.argmax(dim=1).to(prompted_llm.device))
        prompted_start_idx = (torch.arange(0, input_ids.shape[0]).to(prompted_llm.device), prompted_masks_tensor.argmax(dim=1).to(prompted_llm.device))
        

        for i in range(max_len_sequence):
            if unprompted_start_idx[1].max() >= unprompted_logits_peft.shape[1] or prompted_start_idx[1].max() >= prompted_logits_peft.shape[1]:
                # Any of the start indices are out of bounds, remove them from unprompted_start_idx and prompted_start_idx
                unprompted_start_idx = (unprompted_start_idx[0][unprompted_start_idx[1] < unprompted_logits_peft.shape[1]], unprompted_start_idx[1][unprompted_start_idx[1] < unprompted_logits_peft.shape[1]])
                prompted_start_idx = (prompted_start_idx[0][prompted_start_idx[1] < prompted_logits_peft.shape[1]], prompted_start_idx[1][prompted_start_idx[1] < prompted_logits_peft.shape[1]])

                # assert resulting shapes are the same
                assert unprompted_start_idx[0].shape == prompted_start_idx[0].shape
                assert unprompted_start_idx[1].shape == prompted_start_idx[1].shape
                assert unprompted_start_idx[0].shape == unprompted_start_idx[1].shape
                assert prompted_start_idx[0].shape == prompted_start_idx[1].shape

            delta_prime_i_batch = F.kl_div(F.log_softmax(unprompted_logits_peft[unprompted_start_idx], dim=-1),
                                        F.log_softmax(unprompted_logits_base[unprompted_start_idx], dim=-1),
                                        reduction='none',
                                        log_target=True).sum(dim=-1)

            delta_i_batch = F.kl_div(F.log_softmax(prompted_logits_base[prompted_start_idx], dim=-1),
                                F.log_softmax(unprompted_logits_base[unprompted_start_idx], dim=-1),
                                reduction='none',
                                log_target=True).sum(dim=-1)        

            # Mask batch elements where mask is 0
            delta_i_masked = delta_i_batch[unprompted_masks_tensor[unprompted_start_idx] == 1]
            delta_prime_i_masked = delta_prime_i_batch[unprompted_masks_tensor[unprompted_start_idx] == 1]

            if (unprompted_masks_tensor[unprompted_start_idx] == 0).any() and i == 0:
                pdb.set_trace()

            # Append each element of the batch to the main dict
            delta[i] = delta.get(i, []) + delta_i_masked.cpu().tolist()
            delta_prime[i] = delta_prime.get(i, []) + delta_prime_i_masked.cpu().tolist()

            unprompted_start_idx = (unprompted_start_idx[0], unprompted_start_idx[1] + 1)
            prompted_start_idx = (prompted_start_idx[0], prompted_start_idx[1] + 1)

    return delta, delta_prime


def get_dataset_from_args(args):
    if args.data_file == "NONE": 
        args = get_og_args(args.results_dir)
        data_file = args['val_path']
    else: 
        data_file = args.data_file

    dataset = load_dataset('json', data_files=data_file)
    return dataset, data_file


def draw_graphs(res_dict, results_dir, hash=""):
    log("Drawing graphs")
    df = pd.DataFrame(res_dict)

    # UP_peft, P_base
    fig = px.scatter(df, x='unprompted_logits_peft_losses', y='prompted_logits_base_losses', color='lengths')
    # title 
    fig.update_layout(title=f'Unprompted PEFT vs Prompted Base -- {hash}')
    # figure bounds 0-8 
    fig.update_xaxes(range=[0, 8])
    fig.update_yaxes(range=[0, 8])

    # save fig
    fig.write_html(os.path.join(results_dir, f"{hash}_up_peft_vs_p_base.html"))


    # P_peft, P_base
    fig = px.scatter(df, x='prompted_logits_peft_losses', y='prompted_logits_base_losses', color='lengths')
    # title
    fig.update_layout(title=f'Prompted PEFT vs. Prompted Base-- {hash}')
    # figure bounds 0-8 
    fig.update_xaxes(range=[0, 8])
    fig.update_yaxes(range=[0, 8])
    # save fig
    fig.write_html(os.path.join(results_dir, f"{hash}_p_peft_vs_p_base.html"))


    # UP_peft UP_base
    fig = px.scatter(df, x='unprompted_logits_peft_losses', y='unprompted_logits_base_losses', color='lengths')
    # title
    fig.update_layout(title=f'Unprompted PEFT vs Unprompted Base -- {hash}')
    # figure bounds 0-8 
    fig.update_xaxes(range=[0, 8])
    fig.update_yaxes(range=[0, 8])
    # save fig
    fig.write_html(os.path.join(results_dir, f"{hash}_up_peft_vs_up_base.html"))


    # P_peft UP_base
    fig = px.scatter(df, x='prompted_logits_peft_losses', y='unprompted_logits_base_losses', color='lengths')
    # title
    fig.update_layout(title=f'Prompted PEFT vs Unprompted Base -- {hash}')
    # figure bounds 0-8 
    fig.update_xaxes(range=[0, 8])
    fig.update_yaxes(range=[0, 8])
    # save fig
    fig.write_html(os.path.join(results_dir, f"{hash}_p_peft_vs_up_base.html"))


    # P_peft, UP_peft
    fig = px.scatter(df, x='prompted_logits_peft_losses', y='unprompted_logits_peft_losses', color='lengths')
    # title
    fig.update_layout(title=f'Prompted PEFT vs Unprompted PEFT -- {hash}')
    # figure bounds 0-8 
    fig.update_xaxes(range=[0, 8])
    fig.update_yaxes(range=[0, 8])
    # save fig
    fig.write_html(os.path.join(results_dir, f"{hash}_p_peft_vs_up_peft.html"))


    # P_base, UP_base 
    fig = px.scatter(df, x='prompted_logits_base_losses', y='unprompted_logits_base_losses', color='lengths')
    # title
    fig.update_layout(title=f'Prompted Base vs Unprompted Base -- {hash}')
    # figure bounds 0-8 
    fig.update_xaxes(range=[0, 8])
    fig.update_yaxes(range=[0, 8])
    # save fig
    fig.write_html(os.path.join(results_dir, f"{hash}_p_base_vs_up_base.html"))



    log("Done drawing graphs")

def draw_graphs_deviance(prompted_llm, peft_model, tokenizer, dataset, batch_size, results_dir, hash=""):
    log("Computing deviance")
    delta_dict, delta_prime_dict = compute_deviance(prompted_llm, peft_model, tokenizer, dataset, batch_size)
    log("Done computing deviance")


    delta_avg_list = []
    delta_prime_avg_list = []
    for i in range(len(delta_dict)):
        delta_avg_list.append(sum(delta_dict[i]) / len(delta_dict[i]))
        delta_prime_avg_list.append(sum(delta_prime_dict[i]) / len(delta_prime_dict[i]))

    # For delta_prime_avg_list compute a rolling average of 10
    delta_avg_list_rolling = pd.Series(delta_avg_list).rolling(200).mean().tolist()
    delta_prime_avg_list_rolling = pd.Series(delta_prime_avg_list).rolling(200).mean().tolist()

    # Compute delta_rolling / delta_prime_rolling
    delta_divided = [delta_avg_list_rolling[i] / delta_prime_avg_list_rolling[i] for i in range(len(delta_avg_list_rolling))]
    ones = [1 for _ in range(len(delta_avg_list_rolling))]

    # Draw the graph
    df = pd.DataFrame({
        'delta / delta': ones,
        'delta / delta_prime': delta_divided
    })

    # Draw a single plot with two lines on it

    fig = px.line(df, x=df.index, y=['delta / delta', 'delta / delta_prime'])
    fig.update_layout(title=f'Deviance -- {hash}')
    fig.write_html(os.path.join(results_dir, f"{hash}_deviance_invrel.html"))


def main(): 
    # parse args
    args = parse_args()

    # log setup 
    log.log_path = os.path.join(args.results_dir, "compare_models.log")

    # get the dataset
    log(f"Getting dataset from {args.data_file}")
    dataset, dataset_path = get_dataset_from_args(args)
    log("Done getting dataset.")

    # get the md5sum of dataset_path
    log("Building path prefix")
    path_prefix = os.path.splitext(os.path.basename(dataset_path))[0] 
    if args.u_override is not None:
        path_prefix += "_u_override_"+os.path.splitext(os.path.basename(args.u_override))[0]+"_"
    log(f"path_prefix: {path_prefix}")



    # get the models
    log(f"Getting PEFT model from {args.results_dir}")
    base_model, peft_model, tokenizer = get_models(args.results_dir)
    log("Done getting models.")



    # compute the log likelihoods
    log("Computing log likelihoods")
    res_dict = get_loss(base_model, peft_model, tokenizer, dataset, 
                        batch_size=args.batch_size, optimizer=None, do_step=False, 
                        u_override=args.u_override)
    log("Done computing log likelihoods")

    # save the results
    res_path = os.path.join(args.results_dir, f"{path_prefix}_compare_models_results.json")
    log(f"Saving all losses for all questions to {res_path}")
    with open(res_path, 'w') as f: 
        json.dump(res_dict, f)
    log("Done saving results")

    # draw the scatter plots
    draw_graphs(res_dict, args.results_dir, hash=path_prefix)

def main_deviance():
    args = parse_args()
    log.log_path = os.path.join(args.results_dir, "compare_models.log")
    log(f"Getting dataset from {args.data_file}")
    dataset, dataset_path = get_dataset_from_args(args)
    log("Done getting dataset.")

    log(f"Getting models from {args.results_dir}")
    base_model, peft_model, tokenizer = get_models(args.results_dir)
    log("Done getting models.")


    draw_graphs_deviance(base_model, peft_model, tokenizer, dataset, args.batch_size, args.results_dir, hash=os.path.basename(dataset_path))



if __name__ == "__main__": 
    main()
    # main_deviance()