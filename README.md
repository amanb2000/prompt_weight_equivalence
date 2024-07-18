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
# generate trajectory dataset with ground truth prompted logits
python3 generate_data.py \
    --prompt_template data/llama_sys_question_template.md \
    --x0_file data/truth_x0.md \
    --train_questions data/squad_train.jsonl \
    --val_questions data/squad_validation.jsonl \
    --num_sequences_per_question 25 \
    --num_questions 100 \
    --max_sequence_length 300 \
    --min_sequence_length 100 \
    --temperature 2.0 \
    --batch_size 25 \
    --train_out_file data/train_traj_temp2.0_numq100_numseq25_x0truth_20240718.jsonl
    --val_out_file data/val_traj_temp2.0_numq100_numseq25_x0truth_20240718.jsonl
    --model_name meta-llama/Meta-Llama-3-8B-Instruct 

python3 train_loop.py --help
```