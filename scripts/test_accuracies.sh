#!/bin/bash


# Array of model epochs
epochs=(9 19 29 39)

# Loop through each directory in cotqa
for dir in results/ballmer_20240726/cotqa/*/; do
    # Remove trailing slash from directory name
    dir=${dir%/}
    
    # Extract the directory name without the path
    dir_name=$(basename "$dir")
    
    # Loop through each epoch
    for epoch in "${epochs[@]}"; do
        # Run the Python command
        python3 test_math_model.py \
            --results_dir "results/ballmer_20240726/cotqa/${dir_name}" \
            --batch_size 32 \
            --num_questions 400 \
            --model_epoch "$epoch"
        
        echo "Completed run for ${dir_name} with model_epoch ${epoch}"
    done
done