import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import save_json, read_data_summary
from prompt.aug_desc_llama_prompt import generate_prompt
from config import DefaultConfig

default_config = DefaultConfig()
# set display_flag to True to display the prompt for inspection
display_flag = False


def init_llama():
    print("Start loading the model...")
    model = AutoModelForCausalLM.from_pretrained(default_config.llama_model_id,
                                                 torch_dtype=default_config.model_torch_type,
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(default_config.llama_model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        
    model.eval()
    
    return model, tokenizer


def run_llama(model, tokenizer, 
              name, normal_label_list, anomaly_label_list,
              origianl_task):
    prompt = generate_prompt(name, 
                             normal_label_list, anomaly_label_list,
                             origianl_task)
    global display_flag
    if display_flag:
        print("Here is the prompt for inspection:")
        print(prompt)
        display_flag = False
        
    with torch.no_grad():
        input_ = tokenizer(prompt, return_tensors="pt")
        input_.to(default_config.device)
        generated_ids = model.generate(
            **input_,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=default_config.more_max_new_tokens,
            do_sample=True,
            temperature=0.5
        )
    generated_text = tokenizer.decode(
        generated_ids[0], 
        skip_special_tokens=True
    )

    try:
        match = re.search(r'\{.*\}', generated_text, re.DOTALL)
    except Exception as e:
        raise ValueError(f"!!! Match Error: {e}")
    
    if match:
        generated_json = match.group(0)
        print(generated_json)
        generated_dict = json.loads(generated_json)
        # save the generated_json to a file
        save_json(generated_dict, name, "llama_desc")
    else:
        raise ValueError(f"!!! Error: JSON not found in {generated_text}")
    

def main():
    model, tokenizer = init_llama()
    normal_label_list, anomaly_label_list, origianl_task, _ = \
        read_data_summary(default_config.data_name)
    run_llama(model, tokenizer, 
              default_config.data_name, 
              normal_label_list, anomaly_label_list,
              origianl_task)
    

if __name__ == "__main__":
    main()
