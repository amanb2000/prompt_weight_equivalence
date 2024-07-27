"""
Data generation commands for 2024 07 26 Experiments on Chain-of-Though QA (CoT-QA)

Parameters to vary: 
 --x0_file: ["data/cot_math_aqua-mathqa_x0.md", "data/cot_math_x0.md"] - We are using the few-shot learning x0 files from page 35 (table 20) of https://arxiv.org/pdf/2201.11903.  - FOR EACH DATASET, THERE IS ONLY ONE DATASET YOU ACTUALLY USE.
 --question_dataset 
     - 5 datasets, data/*.jsonl, train/valid sets included.
 --temperature [1.0, 2.0, 3.0, 5.0]
 --traj_out_file: based on temperature, question_dataset, x0_file, stored in data/ballmer_20240726


Parameters to keep constant: 
 --num_questions 500
 --num_sequences_per_question 5
 --max_sequence_length 100
 --min_sequence_length 1
 --batch_size 10


**Data generation**: python3 
 - x0_file: `data/*_x0.md` 
 - num_questions: 25 
 - num_sequuences_per_question: 25
 - min_sequence_length, max_sequence_length = (10, 100), (100, 300), (300, 1000)
 - temperatures: 0.5, 1, 3, 5, 10, 20
 - batch_size: 3
 - traj_out_file: Based on names above, stored in `data/20240725`

usage: generate_data.py [-h] [--x0_file X0_FILE] [--question_dataset QUESTION_DATASET] [--num_questions NUM_QUESTIONS]
                        [--num_sequences_per_question NUM_SEQUENCES_PER_QUESTION]
                        [--max_sequence_length MAX_SEQUENCE_LENGTH] [--min_sequence_length MIN_SEQUENCE_LENGTH]
                        [--temperature TEMPERATURE] [--batch_size BATCH_SIZE] [--model_name MODEL_NAME]
                        [--traj_out_file TRAJ_OUT_FILE] [--seed SEED]

Model loading and generation parameters.

options:
  -h, --help            show this help message and exit
  --x0_file X0_FILE     Path to the x0 file. Default = "data/truth_x0.md"
  --question_dataset QUESTION_DATASET
                        Path to the question dataset file. Default = "data/squad_train.jsonl"
  --num_questions NUM_QUESTIONS
                        Number of questions to generate. Default = 100
  --num_sequences_per_question NUM_SEQUENCES_PER_QUESTION
                        Number of sequences to generate per question (default: 25)
  --max_sequence_length MAX_SEQUENCE_LENGTH
                        Maximum length of each generated sequence
  --min_sequence_length MIN_SEQUENCE_LENGTH
                        Minimum length of each generated sequence
  --temperature TEMPERATURE
                        Temperature for sequence generation
  --batch_size BATCH_SIZE
                        Batch size for processing
  --model_name MODEL_NAME
                        Model name to use
  --traj_out_file TRAJ_OUT_FILE
                        Output file for generated sequences. Default = "data/traj_lex.jsonl"
  --seed SEED           Seed for random number generation
"""
import glob
import os
import pdb

# get the list of x0 files from data/*_x0.md
question_dataset__x0_file = [
    ("data/aqua_validation.jsonl", "data/cot_math_aqua-mathqa_x0.md"),
    ("data/aqua_train.jsonl", "data/cot_math_aqua-mathqa_x0.md"),

    ("data/mathqa_validation.jsonl", "data/cot_math_aqua-mathqa_x0.md"),
    ("data/mathqa_train.jsonl", "data/cot_math_aqua-mathqa_x0.md"),

    ("data/asdiv_validation.jsonl", "data/cot_math_x0.md"), 
    ("data/asdiv_train.jsonl", "data/cot_math_x0.md"),

    ("data/gsm8k_validation.jsonl", "data/cot_math_x0.md"), 
    ("data/gsm8k_train.jsonl", "data/cot_math_x0.md"), 

    ("data/svamp_validation.jsonl", "data/cot_math_x0.md"), 
    ("data/svamp_train.jsonl", "data/cot_math_x0.md")
]

temperatures = [1.0, 2.0, 3.0, 5.0]

base_dir = "data/ballmer_20240726"

num_questions = 500
num_sequences_per_question = 5
max_sequence_length = 100
min_sequence_length = 1
batch_size = 10

seed = 238948723



# mkdir -p base_dir
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

cnt = 0
for question_dataset, x0_file in question_dataset__x0_file:
    for temperature in temperatures: 
        # note: we can ignore the x0_file because there is a 1:1 mapping between x0_file and question_dataset
        question_dataset_basename = os.path.splitext(os.path.basename(question_dataset))[0]

        traj_out_file = os.path.join(base_dir, f"traj_{question_dataset_basename}_temperature_{temperature}.jsonl")

        pdb.set_trace()

        data_gen_func_call = f"python3 generate_data.py --x0_file {x0_file} --question_dataset {question_dataset} --num_questions {num_questions} --num_sequences_per_question {num_sequences_per_question} --max_sequence_length {max_sequence_length} --min_sequence_length {min_sequence_length} --temperature {temperature} --batch_size {batch_size} --traj_out_file {traj_out_file} --seed {seed}"
        
        print(f"[{cnt}] This is the data gen script call: ", data_gen_func_call)

        # write to scripts/commands_cotqa_20240726.txt
        with open("scripts/commands_cotqa_20240726.txt", "a") as f:
            f.write(data_gen_func_call + "\n")

        cnt += 1