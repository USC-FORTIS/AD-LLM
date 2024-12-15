import time
import json
import re
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TextDataset, evaluate
import openai
from prompt.ad_1_gpt_prompt import generate_prompt_setting_1
from prompt.ad_2_gpt_prompt import generate_prompt_setting_2
from config import DefaultConfig, PrivacyConfig

default_config = DefaultConfig()
privacy_config = PrivacyConfig()
# set display_flag to True to display the prompt for inspection
display_flag = False


def init_gpt():
    client = openai.OpenAI(
        organization=privacy_config.organization,
        project=privacy_config.project,
        api_key=privacy_config.gpt_api_key
    )
    return client


# request with retry for error handling
def request_with_retry(prompt, gpt_client, 
                       max_retries=20, retry_after=5):
    for _ in range(max_retries):
        try:
            response = gpt_client.chat.completions.create(
                model=default_config.gpt_model_id,
                messages=prompt,
                max_tokens=default_config.max_new_tokens,
                seed=default_config.seed,
                temperature=0,
            )
            return response
        except openai.BadRequestError as e:
            # caused by filtering
            print(f"!!! BadRequstError: {e}")
            break
        except openai.OpenAIError as e:
            # caused by rate limit
            print(f"!!! RateLimitError: {e}. Retry after {retry_after} seconds.")
            time.sleep(retry_after)
            continue
        except Exception as e:
            print(f"!!! Unknown Error: {e}")
            break
    return None


def detect_anomaly(gpt_client, text_batch, 
                   normal_label_list, anomaly_label_list=None, 
                   normal_desc_dict=None, anomaly_desc_dict=None, 
                   origianl_task=None):
    anomaly_score_list = []

    if default_config.ad_setting == 1:
        generate_prompt = generate_prompt_setting_1
    elif default_config.ad_setting == 2:
        generate_prompt = generate_prompt_setting_2
    else:
        raise ValueError("Invalid ad_setting value.")
    
    for text in text_batch:
        prompt = generate_prompt(text, normal_label_list, anomaly_label_list, 
                                 normal_desc_dict, anomaly_desc_dict, origianl_task)
        global display_flag
        if display_flag:
            print("Here is the first prompt for inspection:")
            print(prompt[0]["content"])
            display_flag = False
        response = request_with_retry(prompt, gpt_client)
        if response is None:
            print(f"!!! Error: No response for text: {text}")
            anomaly_score_list.append(default_config.error_symbol)
            continue

        generated_text = response.choices[0].message.content

        # extract the JSON string using regex
        try:
            match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        except Exception as e:
            print(f"!!! Match Error: {e}")
            print(f"For text: {text}")
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


def run_gpt(data_loader, gpt_client, 
            normal_label_list, anomaly_label_list=None, 
            normal_desc_dict=None, anomaly_desc_dict=None, 
            origianl_task=None):
    y_score = []

    for text_batch in tqdm(data_loader):
        anomaly_score_list = detect_anomaly(gpt_client, text_batch, 
                                            normal_label_list, anomaly_label_list, 
                                            normal_desc_dict, anomaly_desc_dict, 
                                            origianl_task)
        y_score.extend(anomaly_score_list)

    return y_score


def main():
    gpt_client = init_gpt()
    dataset = TextDataset(default_config.data_name)
    data_loader = DataLoader(dataset, batch_size=default_config.batch_size, 
                             shuffle=False, drop_last=False)
    normal_desc_dict, anomaly_desc_dict = None, None
    if default_config._use_desc:
        normal_desc_dict = dataset.get_normal_desc_dict()
        anomaly_desc_dict = dataset.get_anomaly_desc_dict()

    y_score = run_gpt(data_loader, gpt_client,
                      normal_label_list=dataset.get_normal_label_list(),
                      anomaly_label_list=dataset.get_anomaly_label_list(),
                      origianl_task=dataset.get_origianl_task(),
                      normal_desc_dict=normal_desc_dict,
                      anomaly_desc_dict=anomaly_desc_dict)
    y_true = dataset.get_labels()
    evaluate(y_true, y_score)


if __name__ == "__main__":
    main()
