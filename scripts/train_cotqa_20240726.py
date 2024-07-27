import glob
import os
import pdb


traj_train_traj_validation = [("data/ballmer_20240726/traj_asdiv_train_temperature_1.0.jsonl",
                               "data/ballmer_20240726/traj_asdiv_validation_temperature_1.0.jsonl"),

                               ("data/ballmer_20240726/traj_asdiv_train_temperature_2.0.jsonl",
                                "data/ballmer_20240726/traj_asdiv_validation_temperature_2.0.jsonl"),

                                ("data/ballmer_20240726/traj_asdiv_train_temperature_3.0.jsonl",
                                 "data/ballmer_20240726/traj_asdiv_validation_temperature_3.0.jsonl"),
                                 
                                 ("data/ballmer_20240726/traj_asdiv_train_temperature_5.0.jsonl",
                                  "data/ballmer_20240726/traj_asdiv_validation_temperature_5.0.jsonl"),
                                  
                                  ("data/ballmer_20240726/traj_aqua_train_temperature_1.0.jsonl",
                                   "data/ballmer_20240726/traj_aqua_validation_temperature_1.0.jsonl"),
                                   
                                   ("data/ballmer_20240726/traj_aqua_train_temperature_2.0.jsonl",
                                    "data/ballmer_20240726/traj_aqua_validation_temperature_2.0.jsonl"),
                                    
                                    ("data/ballmer_20240726/traj_aqua_train_temperature_3.0.jsonl",
                                     "data/ballmer_20240726/traj_aqua_validation_temperature_3.0.jsonl"),
                                     
                                     ("data/ballmer_20240726/traj_aqua_train_temperature_5.0.jsonl",
                                      "data/ballmer_20240726/traj_aqua_validation_temperature_5.0.jsonl"),
                                      
                                      ("data/ballmer_20240726/traj_gsm8k_train_temperature_1.0.jsonl",
                                       "data/ballmer_20240726/traj_gsm8k_validation_temperature_1.0.jsonl"),
                                       
                                       ("data/ballmer_20240726/traj_gsm8k_train_temperature_2.0.jsonl",
                                        "data/ballmer_20240726/traj_gsm8k_validation_temperature_2.0.jsonl"),
                                        
                                        ("data/ballmer_20240726/traj_gsm8k_train_temperature_3.0.jsonl",
                                         "data/ballmer_20240726/traj_gsm8k_validation_temperature_3.0.jsonl"),
                                         
                                         ("data/ballmer_20240726/traj_gsm8k_train_temperature_5.0.jsonl",
                                          "data/ballmer_20240726/traj_gsm8k_validation_temperature_5.0.jsonl"),
                                          
                                          ("data/ballmer_20240726/traj_svamp_train_temperature_1.0.jsonl",
                                           "data/ballmer_20240726/traj_svamp_validation_temperature_1.0.jsonl"),
                                           
                                           ("data/ballmer_20240726/traj_svamp_train_temperature_2.0.jsonl",
                                            "data/ballmer_20240726/traj_svamp_validation_temperature_2.0.jsonl"),
                                            
                                            ("data/ballmer_20240726/traj_svamp_train_temperature_3.0.jsonl",
                                             "data/ballmer_20240726/traj_svamp_validation_temperature_3.0.jsonl"),
                                             
                                             ("data/ballmer_20240726/traj_svamp_train_temperature_5.0.jsonl",
                                              "data/ballmer_20240726/traj_svamp_validation_temperature_5.0.jsonl"),
                                              
                                              ("data/ballmer_20240726/traj_mathqa_train_temperature_1.0.jsonl",
                                               "data/ballmer_20240726/traj_mathqa_validation_temperature_1.0.jsonl"),
                                               
                                               ("data/ballmer_20240726/traj_mathqa_train_temperature_2.0.jsonl",
                                                "data/ballmer_20240726/traj_mathqa_validation_temperature_2.0.jsonl"),
                                                
                                                ("data/ballmer_20240726/traj_mathqa_train_temperature_3.0.jsonl",
                                                 "data/ballmer_20240726/traj_mathqa_validation_temperature_3.0.jsonl"),
                                                 
                                                 ("data/ballmer_20240726/traj_mathqa_train_temperature_5.0.jsonl",
                                                  "data/ballmer_20240726/traj_mathqa_validation_temperature_5.0.jsonl")]

lrs = (3e-4, 1e-4)
rs = (1, 2, 4)
batch_size = 16
base_dir = "data/ballmer_20240726"
results_dir = "results/ballmer_20240726"
num_epochs = 40


# mkdir -p base_dir
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

cnt = 0
for train_dataset, validation_dataset in traj_train_traj_validation:
    for lr in lrs:
        for r in rs:
            # note: we can ignore the x0_file because there is a 1:1 mapping between x0_file and question_dataset
            question_dataset_basename = os.path.splitext(os.path.basename(train_dataset))[0]
            
            folder_name = question_dataset_basename.split("traj_")[1]
            out_dir = f"{results_dir}/cotqa/{folder_name}_batch_{batch_size}_epochs_{num_epochs}_lr_{lr}_r_{r}"

            data_gen_func_call = f"python3 train_loop.py --num_epochs {num_epochs} --batch_size {batch_size} -r {r} --learning_rate {lr} --data_path {train_dataset} --val_path {validation_dataset} --out_dir {out_dir}"
            
            print(f"[{cnt}] This is the data gen script call: ", data_gen_func_call)

            # write to scripts/commands_cotqa_20240726.txt
            with open("scripts/commands_TRAIN_cotqa_20240726.txt", "a") as f:
                f.write(data_gen_func_call + "\n")

            cnt += 1
