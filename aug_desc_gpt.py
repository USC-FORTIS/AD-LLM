import time
import json
import re
from utils import save_json, read_data_summary
import openai
from prompt.aug_desc_gpt_prompt import generate_prompt
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
def request_with_retry(prompt, gpt_client, max_retries=20):
    for _ in range(max_retries):
        try:
            response = gpt_client.chat.completions.create(
                model=default_config.gpt_model_id,
                messages=prompt,
                max_tokens=default_config.more_max_new_tokens,
                seed=default_config.seed,
                temperature=0.5,
            )
            return response
        except openai.BadRequestError as e:
            # Caused by filtering
            print(f"!!! BadRequstError: {e}")
            break
        except openai.OpenAIError as e:
            retry_after = 5
            print(f"!!! RateLimitError: {e}. Retry after {retry_after} seconds.")
            time.sleep(retry_after)
            continue
        except Exception as e:
            print(f"!!! Unknown Error: {e}")
            break
    return None


def run_gpt(gpt_client, name,
            normal_label_list, anomaly_label_list,
            origianl_task):
    prompt = generate_prompt(name,
                             normal_label_list, anomaly_label_list,
                             origianl_task)
    response = request_with_retry(prompt, gpt_client)

    if response is None:
        raise ValueError("!!! Error: response is None.")
    
    generated_text = response.choices[0].message.content

    # extract the JSON string using regex
    try:
        match = re.search(r'\{.*\}', generated_text, re.DOTALL)
    except Exception as e:
        raise ValueError(f"!!! Match Error: {e}")
    
    if match:
        generated_json = match.group(0)
        print(generated_json)
        generated_dict = json.loads(generated_json)
        # save the generated_json to a file
        save_json(generated_dict, name, "gpt_desc")
    else:
        raise ValueError(f"!!! Error: JSON not found in {generated_text}")
    
    
def main():
    gpt_client = init_gpt()
    normal_label_list, anomaly_label_list, origianl_task, _ = \
        read_data_summary(default_config.data_name)
    run_gpt(gpt_client, 
            default_config.data_name, 
            normal_label_list, anomaly_label_list,
            origianl_task)
    

if __name__ == "__main__":
    main()
    