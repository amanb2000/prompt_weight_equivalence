# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse
import pdb
import torch


parser = argparse.ArgumentParser(description="Model loading and generation parameters.")
# arg: prompt_template, defaults to data/llama_sys_question_template.md
parser.add_argument('--prompt_template', type=str, default="data/llama_sys_question_template.md", help='Path to the prompt template file. Must have two {} instances, first for system prompt, second for question. Default = "data/llama_sys_question_template.md"')
# arg: x0_file, defaults to data/truth_x0.md
parser.add_argument('--x0_file', type=str, default="data/truth_x0.md", help='Path to the x0 file. Default = "data/truth_x0.md"')
# arg: question_dataset_file, default data/squad_train.jsonl
parser.add_argument('--train_questions', type=str, default="data/squad_train.jsonl", help='Path to the question dataset file. Default = "data/squad_train.jsonl"')
# arg: val_questions, default data/squad_val.jsonl
parser.add_argument('--val_questions', type=str, default="data/squad_validation.jsonl", help='Path to the validation question dataset file. Default = "data/squad_val.jsonl"')
# arg:num_questions, default = 100
parser.add_argument('--num_questions', type=int, default=100, help='Number of questions to generate. Default = 100')

parser.add_argument('--num_sequences_per_question', type=int, default=25, help='Number of sequences to generate per question (default: 25)')
parser.add_argument('--max_sequence_length', type=int, default=300, help='Maximum length of each generated sequence')
parser.add_argument('--min_sequence_length', type=int, default=100, help='Minimum length of each generated sequence')
parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for sequence generation')
parser.add_argument('--batch_size', type=int, default=38, help='Batch size for processing')
parser.add_argument('--question_dataset', type=str, default="DATASET", help='Question dataset to use')
parser.add_argument('--model_name', type=str, default= "meta-llama/Meta-Llama-3-8B-Instruct", help='Model name to use')
parser.add_argument('--train_out_file', type=str, default="data/traj_lex.jsonl", help='Output file for generated sequences. Default = "data/traj_lex.jsonl"')
parser.add_argument('--val_out_file', type=str, default="data/traj_lex_val.jsonl", help='Output file for generated sequences. Default = "data/traj_lex_val.jsonl"')

args = parser.parse_args()



def format_prompt(system_prompt, user_prompt):
    return f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}\n\n<|start_header_id|>user<|end_header_id|>{user_prompt}<|start_header_id|>assistant<|end_header_id|>"

# load prompt_template from args.prompt_template 
with open(args.prompt_template, 'r') as f:
    prompt_template = f.read()
print("Prompt_template: ", prompt_template)

# load x0_file from args.x0_file
with open(args.x0_file, 'r') as f:
    x0_str = f.read()

# load question_dataset_file from args.question_dataset_file
import datasets
train_dataset = datasets.load_dataset('json', data_files=args.train_questions)
print("Length of train_dataset: ", len(train_dataset['train']))

val_dataset = datasets.load_dataset('json', data_files=args.val_questions)
print("Length of val_dataset: ", len(val_dataset['train']))







tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").half()
tokenizer.pad_token = tokenizer.eos_token

model = model.cuda()

num_q = 0


with open(args.train_out_file, 'w') as f: 
    pbar = tqdm(train_dataset['train'], total=args.num_questions)
    for question in pbar: 
        if num_q >= args.num_questions:
            break
        num_q += 1
        question_str = question['question']
        for i in range(0, args.num_sequences_per_question, args.batch_size): 
            # print("Generating sequences for question: ", question)
            batch_start = i
            batch_end = min(i+args.batch_size, args.num_sequences_per_question)
            # print(f"Batch: {batch_start}:{batch_end}")
            prompt_q_str = prompt_template.format(x0_str, question_str)

            num_seq_to_gen = batch_end - batch_start

            input_ids = tokenizer(prompt_q_str, return_tensors="pt").to(model.device)

            batch_input_ids = input_ids['input_ids'].repeat(num_seq_to_gen, 1)
            attention_mask = batch_input_ids.ne(tokenizer.pad_token_id).long()

            # output has shape [batch, num_tokens_total]
            output = model.generate(
                batch_input_ids, 
                attention_mask = attention_mask,
                do_sample = True, 
                max_new_tokens = args.max_sequence_length,
                min_length = args.min_sequence_length,
                temperature = args.temperature,
                pad_token_id = tokenizer.eos_token_id
            )

            # attention_mask = output.ne(tokenizer.pad_token_id).long()
            # with torch.no_grad(): 
            #     inf_out = model(output, attention_mask = attention_mask)
            # batch_logits = inf_out.logits # [batch, num_toks, vocab_size]

            for j in range(num_seq_to_gen): 
                # Decode the generated sequence
                generated_text = tokenizer.decode(output[j], skip_special_tokens=False)
                # print(f"Generated text i={i} j={j}: ", generated_text)

                # create a dictionary with the required format 
                # make an attention mask for the input_ids -- no attention to pad tokens 

                example = {
                    "text": generated_text, 
                    "input_ids": output[j].tolist(), 
                    "attention_mask": attention_mask[j].tolist()
                    # "logits": batch_logits[j, :, :].tolist()
                }

                # writ ethe example to the jsonl file 
                json.dump(example, f)
                f.write("\n")

print("Dataset saved to ", args.train_out_file)