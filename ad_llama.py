import json
import re
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import TextDataset, evaluate
from prompt.ad_1_llama_prompt import generate_prompt_setting_1
from prompt.ad_2_llama_prompt import generate_prompt_setting_2
from config import DefaultConfig

default_config = DefaultConfig()
# set display_flag to True to display the prompt for inspection
display_flag = False


def init_llama():
    print("Start loading the model...")
    model = AutoModelForCausalLM.from_pretrained(default_config.llama_model_id,
                                                 torch_dtype=default_config.model_torch_type,
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(default_config.llama_model_id,
                                              padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    model.eval()

    return model, tokenizer


def detect_anomaly(model, tokenizer, 
                   text_batch, normal_label_list, 
                   anomaly_label_list=None, 
                   normal_desc_dict=None, anomaly_desc_dict=None, 
                   origianl_task=None):
    anomaly_score_list = []

    if default_config.ad_setting == 1:
        generate_prompt = generate_prompt_setting_1
    elif default_config.ad_setting == 2:
        generate_prompt = generate_prompt_setting_2
    else:
        raise ValueError("Invalid ad_setting value.")
    
    prompt_batch = [generate_prompt(text, normal_label_list, anomaly_label_list, 
                                     normal_desc_dict, anomaly_desc_dict, origianl_task) 
                    for text in text_batch]
    
    global display_flag
    if display_flag:
        print("Here is the first prompt for inspection:")
        print(prompt_batch[0])
        display_flag = False

    with torch.no_grad():
        input_batch = tokenizer(prompt_batch, return_tensors="pt", padding=True)
        input_batch.to(default_config.device)
        generated_ids_batch = model.generate(
            **input_batch,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=default_config.max_new_tokens,
            do_sample=False
        )
    generated_text_batch = tokenizer.batch_decode(
        generated_ids_batch, 
        skip_special_tokens=True
    )

    for generated_text in generated_text_batch:
        # extract the JSON string using regex
        try:
            match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        except Exception as e:
            print(f"!!! Match Error: {e}")
            anomaly_score_list.append(default_config.error_symbol)
            continue
        
        if match:
            generated_json = match.group(0)
            try:
                generated_dict = json.loads(generated_json)
                anomaly_score = generated_dict["anomaly_score"]

                # transform the anomaly_score to float
                anomaly_score = float(anomaly_score)
                    
                # check if the anomaly_score is out of range
                if anomaly_score < 0 or anomaly_score > 1:
                    print(f"!!! Error: anomaly_score out of range: {anomaly_score}")
                    anomaly_score = default_config.error_symbol

                anomaly_score_list.append(anomaly_score)

            except json.JSONDecodeError as e:
                print(f"!!! Error decoding JSON: {e}, for text: {generated_json}")
                anomaly_score_list.append(default_config.error_symbol)
        else:
            print(f"!!! Error: JSON not found in {generated_text}")
            anomaly_score_list.append(default_config.error_symbol)

    return anomaly_score_list


def run_llama(data_loader, model, tokenizer, 
              normal_label_list, anomaly_label_list=None, 
              normal_desc_dict=None, anomaly_desc_dict=None, 
              origianl_task=None):
    y_score = []
    for text_batch in tqdm(data_loader):
        anomaly_score_list = detect_anomaly(model, tokenizer, text_batch, 
                                            normal_label_list, anomaly_label_list, 
                                            normal_desc_dict, anomaly_desc_dict, 
                                            origianl_task)
        y_score.extend(anomaly_score_list)
    return y_score


def main():
    model, tokenizer = init_llama()
    dataset = TextDataset(default_config.data_name, model_name="llama")
    data_loader = DataLoader(dataset, batch_size=default_config.batch_size, 
                             shuffle=False, drop_last=False)
    normal_desc_dict, anomaly_desc_dict = None, None
    if default_config.use_desc:
        normal_desc_dict = dataset.get_normal_desc_dict()
        anomaly_desc_dict = dataset.get_anomaly_desc_dict()

    y_score = run_llama(data_loader, model, tokenizer, 
                        normal_label_list=dataset.get_normal_label_list(),
                        anomaly_label_list=dataset.get_anomaly_label_list(),
                        origianl_task=dataset.get_origianl_task(),
                        normal_desc_dict=normal_desc_dict,
                        anomaly_desc_dict=anomaly_desc_dict)
    y_true = dataset.get_labels()
    evaluate(y_true, y_score)


if __name__ == "__main__":
    main()
