
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse


parser = argparse.ArgumentParser(description="Model loading and generation parameters.")
parser.add_argument('--num_sequences_per_question', type=int, default=1000, help='Number of sequences to generate per question')
parser.add_argument('--max_sequence_length', type=int, default=300, help='Maximum length of each generated sequence')
parser.add_argument('--min_sequence_length', type=int, default=100, help='Minimum length of each generated sequence')
parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for sequence generation')
parser.add_argument('--batch_size', type=int, default=38, help='Batch size for processing')
parser.add_argument('--question_dataset', type=str, default="DATASET", help='Question dataset to use')
parser.add_argument('--model_name', type=str, default= "meta-llama/Meta-Llama-3-8B-Instruct", help='Model name to use')
parser.add_argument('--system_prompt', type=str, default="You love talking about cats and will turn any conversation into a discussion about cats.", help='System prompt to use')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name).half()

model = model.cuda()





def format_prompt(system_prompt, user_prompt):
    return f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}\n\n<|start_header_id|>user<|end_header_id|>{user_prompt}<|start_header_id|>assistant<|end_header_id|>"




traj_file = f"../data/traj_lex_nseq{num_sequences}_maxlen{max_sequence_length}_minlen{min_sequence_length}_temp{temperature}.jsonl"
x0_file = f"../data/x0_lex.txt"






