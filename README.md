# Prompt-Weight Equivalence 

**Goal**: How easy is it to train an LLM via weight updates s.t. its probability
distribution over subsequent token sequences is identical to that of a prompted
model.


## Setup
```bash
# make virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip3 install -r requirements.txt

# download SQuAD dataset into data/*.jsonl
mkdir -p data
python3 download_data.py 
```


## Run Experiments 

```bash
# (TODO) generate trajectory dataset with ground truth prompted logits
python3 generate_data.py \
    --prompt_template data/llama_sys_question_template.md \
    --x0_file data/truth_x0.md \
    --train_questions data/squad_train.jsonl \
    --num_sequences_per_question 25 \
    --num_questions 100 \
    --max_sequence_length 300 \
    --min_sequence_length 100 \
    --temperature 2.0 \
    --batch_size 25 \
    --train_out_file data/train_traj_temp2.0_numq100_numseq25_x0truth_20240718.jsonl
    --model_name meta-llama/Meta-Llama-3-8B-Instruct 

# (TODO) generate validation data
python3 generate_data.py \
    --prompt_template data/llama_sys_question_template.md \
    --x0_file data/truth_x0.md \
    --train_questions data/squad_validation.jsonl \
    --num_sequences_per_question 25 \
    --num_questions 25 \
    --max_sequence_length 300 \
    --min_sequence_length 100 \
    --temperature 2.0 \
    --batch_size 25 \
    --train_out_file data/val_traj_temp2.0_numq25_numseq25_x0truth_20240718.jsonl \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct 



python3 train_loop.py --num_epochs 10 \
    --batch_size 20 \
    --learning_rate 1e-4 \
    --data_path data/train_traj_temp2.0_numq100_numseq25_x0truth_20240718.jsonl \
    --val_path data/val_traj_temp2.0_numq25_numseq25_x0truth_20240718.jsonl \
    --out_dir results/truthful_squad_match_01_ep1000_batch20
```


## Experiments 2024 07 19

### I'm blue baabaaadeeeebuuuuddaa
```bash
# Generate new dataset with improved system prompt management
# one prompt with all caps, one prompt always mention the color blue in each
# sentence you generate
python3 generate_data.py \
    --x0_file data/blue_x0.md \
    --question_dataset data/squad_train.jsonl \
    --num_questions 100 \
    --num_sequences_per_question 25 \
    --max_sequence_length 300 \
    --min_sequence_length 100 \
    --temperature 2.0 \
    --batch_size 64 \
    --traj_out_file data/traj_bluex0_squad_train.jsonl

# generate validation set
python3 generate_data.py \
    --x0_file data/blue_x0.md \
    --question_dataset data/squad_validation.jsonl \
    --num_questions 25 \
    --num_sequences_per_question 25 \
    --max_sequence_length 300 \
    --min_sequence_length 100 \
    --temperature 2.0 \
    --batch_size 32 \
    --traj_out_file data/traj_bluex0_squad_val.jsonl

# Train blue model 
python3 train_loop.py \
    --num_epochs 20 \
    --learning_rate 3e-4 \
    --data_path data/traj_bluex0_squad_train.jsonl \
    --val_path data/traj_bluex0_squad_val.jsonl \
    --out_dir results/blue20240719.1
    --batch_size 32
```











