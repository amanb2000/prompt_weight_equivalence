# Goal: init model, run lora to finetune weights, save model


import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F


# Llama 3 system prompt + user prompt
PROMPT_STRING = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\
\
You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\
\
What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
num_epochs = 100
batch_size = 32
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO Load dataset here
dataset = []


if __name__ == "__main__":
    # Load model
    
    # Initialize a tokenizer and model
    model_name = ""

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

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=1,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "query_key_value"
        ]
    )
    
    peft_model = get_peft_model(model, peft_config)
    print("unprompted model parameter stats:")
    peft_model.print_trainable_parameters()


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
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            prompt_tokenized = tokenizer(PROMPT_STRING, return_tensors="pt", padding=True, truncation=True)
            
            inputs_unprompted = inputs["input_ids"].to(device)
            print("Unprompted inputs shape", inputs_unprompted.shape)
            # Concatenate the prompt to the input for the prompted model for all inputs in the batch
            inputs_prompted = torch.cat([inputs_unprompted.unsqueeze(0).repeat(inputs_prompted.shape[1], 1), inputs_unprompted], dim=1)
            print("Prompted inputs shape", inputs_prompted.shape)

            unprompted_logits = peft_model(inputs_unprompted).logits
            prompted_logits = prompted_llm(inputs_prompted).logits[-inputs_unprompted.shape[1]:]
            print("Unprompted logits shape", unprompted_logits.shape)
            print("Prompted logits shape", prompted_logits.shape)

            # Compute KL divergence: KL(prompted, unprompted)
            kl_div = F.kl_div(F.log_softmax(unprompted_logits, dim=-1), 
                            F.log_softmax(prompted_logits, dim=-1), 
                            reduction='batchmean', 
                            log_target=True)
            
            kl_div.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch} loss: {kl_div.item()}")