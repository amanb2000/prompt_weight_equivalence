import streamlit as st

import os
import time
import socket
import uuid
from datetime import datetime
import hashlib
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# Function to get the client's IP address
def get_client_ip():
    return socket.gethostbyname(socket.gethostname())

# Function to get or create a session ID
def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def log(msg, file_path="dashboard/app.log"): 
    with open(file_path, 'a') as f:
        f.write(f"[{datetime.now()}] {msg}\n")



# Main Streamlit app
def main():
    st.title("Prompt-Weight Equivalence Results")
    st.write("""
    This is a simple Streamlit app to show the results of the prompt-weight equivalence study.
             
     - The files of interest are in `RESULTS_BASE = /mnt/arc/archive/pwe/cam/results/ballmer_20240726/cotqa/`. 
     - For each results folder in `RESULTS_BASE/*/`, there are currently 4 epochs tested `mathest*.json`.
         - Let's group together `RESULTS_BASE/*/` by dataset -- either `["RESULTS_BASE/asdiv*/", "RESULTS_BASE/gsm8k*/", "RESULTS_BASE/svamp*/]`. There will be a dropdown menu for this at the top of this section. 
    """)

    RESULTS_BASE = "/mnt/arc/archive/pwe/cam/results/ballmer_20240726/cotqa"
    dataset_names = ["asdiv", "gsm8k", "svamp"]
    # for each dropdown options, glob the directories and get the mathest*.json files
    # for each mathest*.json file, load the json and get the results
    # for each results, get the prompt, the weight, and the accuracy
    # create a dataframe with these columns

    result_folder_list = {}
    all_results_folders = []
    for option in dataset_names:
        glob_string = os.path.join(RESULTS_BASE, f"{option}*")
        # st.write("Glob string: ", glob_string)
        result_folder_list[f"{option}"] = glob.glob(glob_string)
        all_results_folders.extend(result_folder_list[f"{option}"])

    # find the unique learning rates 
    lrs = []
    for folder_name in all_results_folders:
        # find the learning rate by matching the substring "lr_{LR_FLOAT_VALUE}" in the folder name
        lr = float(folder_name.split("lr_")[1].split("_")[0])
        lrs.append(lr)
    lrs = list(set(lrs))
    lrs.sort()
    # st.write("Learning rates: ", lrs)
    
    # st.write(result_folder_list)
    st.write("## CoT Math Dataset Results")
    # make a dropdown menu for the dataset 
    dataset_option = st.selectbox("Select a dataset", dataset_names)
    # make a dropdown menu for the learning rate
    lr_option = st.selectbox("Select a learning rate", lrs)
    
    # now we need to filter the results folders by the dataset and learning rate
    filtered_results_folders = []
    for folder_name in result_folder_list[dataset_option]:
        lr = float(folder_name.split("lr_")[1].split("_")[0])
        if lr == lr_option:
            filtered_results_folders.append(folder_name)
        
    # st.write("Filtered results folders: ", filtered_results_folders)

    # for each folder in filtered_results, there are 4 data files, one for each 
    # epoch tested. They are of the form mathtest_ep19_numq400_gentemp0.0_datasetnameasdiv.json. 
    # we can find them by globbing for mathtest*.json in the folder. 
    
    filtered_results_to_json = {}
    for folder_name in filtered_results_folders:
        json_files = glob.glob(os.path.join(folder_name, "mathtest*.json"))
        filtered_results_to_json[folder_name] = {} 
        for json_file in json_files:
            # find the epoch number. 
            epoch_num = json_file.split("_ep")[1].split("_")[0]
            # read the json file 
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            filtered_results_to_json[folder_name][epoch_num] = json_data

    # st.write("Filtered results to json: ", filtered_results_to_json)


    # now we will construct a dataframe with columns for the folder name (extracting the temperature, learning rate, rank r as columns too), the epoch number, and "mean_accuracy_base", "mean_accuracy_peft_sys", "mean_accuracy_upper_bound_base", "mean_accuracy_upper_bound_peft_sys", "mean_accuracy_last_sentence_base", "mean_accuracy_last_sentence_peft_sys", ...

    df = pd.DataFrame(columns=["folder_name", "temperature", "lr", "rank_r", "epoch_num", "mean_accuracy_base", "mean_accuracy_peft", "mean_accuracy_peft_sys", "mean_accuracy_upper_bound_base", "mean_accuracy_upper_bound_peft", "mean_accuracy_upper_bound_peft_sys", "mean_accuracy_last_sentence_base", "mean_accuracy_last_sentence_peft", "mean_accuracy_last_sentence_peft_sys"])

    # Recall that the folder names are of the form results/ballmer_20240726/cotqa/asdiv_train_temperature_2.0_lr_0.0001_r_2
    # TODO: build the dataframe from the loaded data. 
    # Construct the dataframe
    rows = []
    for folder_name, epoch_data in filtered_results_to_json.items():
        # Extract temperature, lr, and rank_r from folder name
        temp = float(folder_name.split("temperature_")[1].split("_")[0])
        lr = float(folder_name.split("lr_")[1].split("_")[0])
        rank_r = int(folder_name.split("_r_")[1].split("_")[0])
        
        for epoch_num, json_data in epoch_data.items():
            row = {
                "folder_name": folder_name,
                "temperature": temp,
                "lr": lr,
                "rank_r": rank_r,
                "epoch_num": int(epoch_num),
                "mean_accuracy_base": json_data.get("mean_accuracy_base", None),
                "mean_accuracy_peft": json_data.get("mean_accuracy_peft", None),
                "mean_accuracy_peft_sys": json_data.get("mean_accuracy_peft_sys", None),
                "mean_accuracy_upper_bound_base": json_data.get("mean_accuracy_upper_bound_base", None),
                "mean_accuracy_upper_bound_peft": json_data.get("mean_accuracy_upper_bound_peft", None), 
                "mean_accuracy_upper_bound_peft_sys": json_data.get("mean_accuracy_upper_bound_peft_sys", None),
                "mean_accuracy_last_sentence_base": json_data.get("mean_accuracy_last_sentence_base", None),
                "mean_accuracy_last_sentence_peft": json_data.get("mean_accuracy_last_sentence_peft", None),
                "mean_accuracy_last_sentence_peft_sys": json_data.get("mean_accuracy_last_sentence_peft_sys", None)
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)

    # Display the dataframe
    st.write("### Results DataFrame")
    st.dataframe(df)

    # Optionally, you can add some visualizations here
    st.write("### Accuracy Comparison")
    fig, ax = plt.subplots()
    df.plot(x="epoch_num", y=["mean_accuracy_base", "mean_accuracy_peft_sys"], ax=ax)
    plt.title(f"Accuracy Comparison for {dataset_option} (LR: {lr_option})")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Accuracy")
    st.pyplot(fig)


if __name__ == "__main__":
    main()

