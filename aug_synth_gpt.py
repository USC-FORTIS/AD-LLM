import os
import time
import json
import re
from tqdm import tqdm
import openai
from utils import save_json, read_data_summary, read_normal_desc, read_json
from prompt.aug_synth_keyword_gpt_prompt import generate_keywords_prompt
from prompt.aug_synth_sample_gpt_prompt import generate_sample_with_keywords_only_prompt
from config import DefaultConfig, PrivacyConfig

default_config = DefaultConfig()
privacy_config = PrivacyConfig()
# set to True to display the prompt for inspection
keyword_display_flag = False
sample_display_flag = False


def init_gpt():
    client = openai.OpenAI(
        organization=privacy_config.organization,
        project=privacy_config.project,
        api_key=privacy_config.gpt_api_key
    )
    return client


def generate_keywords_gpt(gpt_client, name, original_task, normal_label_list, 
                          normal_desc_dict=None, num_keyword_groups=50):
    prompt = generate_keywords_prompt(name, original_task, normal_label_list, 
                                      normal_desc_dict, num_keyword_groups)
    global keyword_display_flag
    if keyword_display_flag:
        print("Here is the keyword prompt for inspection:")
        print(prompt[0]["content"])
        print(prompt[1]["content"])
        keyword_display_flag = False
    try:
        response = gpt_client.chat.completions.create(
            model=default_config.gpt_model_id,
            messages=prompt,
            max_tokens=default_config.more_max_new_tokens,
            seed=default_config.seed,
            temperature=1.0
        )
    except openai.BadRequestError as e:
        raise ValueError(f"!!! BadRequstError: {e}")
    except openai.OpenAIError as e:
        raise ValueError(f"!!! RateLimitError: {e}")
    except Exception as e:
        raise ValueError(f"!!! Unknown Error: {e}")
    
    generated_text = response.choices[0].message.content
    print(generated_text)

    try:
        match = re.search(r'\{.*\}', generated_text, re.DOTALL)
    except Exception as e:
        raise ValueError(f"!!! Match Error: {e})")
    
    use_desc = ""
    if default_config._use_desc:
        use_desc = "_use_desc"

    if match:
        generated_json = match.group(0)
        print(generated_json)
        generated_dict = json.loads(generated_json)
        # save the generated_json to a file
        save_json(generated_dict, name, f"gpt_keywords{use_desc}")
    else:
        raise ValueError(f"!!! Error: JSON not found in {generated_text}")
    

def generate_sample_with_keywords_gpt(gpt_client, name, original_task,
                                      max_retries=20, retry_after=5):
    # read keywords from the file
    use_desc = ""
    if default_config._use_desc:
        use_desc = "_use_desc"
    keywords_file = f"{name}_gpt_keywords{use_desc}.json"
    keywords_dict = read_json(name, keywords_file)

    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, 'data')
    output_file = os.path.join(data_dir, name, f"{name}_gpt_synth_data{use_desc}.jsonl")

    # get the total number of iterations for progress bar (total keyword groups)
    total_keyword_groups = 0
    for category, keywords in keywords_dict.items():
        group_size = len(keywords)
        if group_size < default_config.num_keyword_groups_act:
            total_keyword_groups += group_size
            print(f"Category: {category}, keywords count: {group_size}")
        else:
            total_keyword_groups += default_config.num_keyword_groups_act
            print(f"Category: {category}, keywords count: {default_config.num_keyword_groups_act}")

    with open(output_file, 'w') as jsonl_file:
        # use tqdm for the progress bar
        with tqdm(total=total_keyword_groups, desc="Generating samples") as pbar:
            # iterate over categories and keywords
            for category, keywords in keywords_dict.items():
                count = 0
                for keyword in keywords:
                    # generate the prompt for the current set of keywords
                    prompt = generate_sample_with_keywords_only_prompt(
                        name, original_task, category, keyword)
                    for _ in range(max_retries):
                        try:
                            response = gpt_client.chat.completions.create(
                                model=default_config.gpt_model_id,
                                messages=prompt,
                                max_tokens=default_config.max_new_tokens,
                                seed=default_config.seed,
                                temperature=1.0
                            )
                            break
                        except openai.BadRequestError as e:
                            # caused by filtering
                            print(f"!!! BadRequstError: {e}")
                            break
                        except openai.OpenAIError as e:
                            print(f"!!! RateLimitError: {e}. Retry after {retry_after} seconds.")
                            time.sleep(retry_after)
                            continue
                        except Exception as e:
                            print(f"!!! Unknown Error: {e}")
                            break
                    
                    generated_text = response.choices[0].message.content
                    # remove the outermost quotes using regular expression
                    generated_text = re.sub(r'^"(.*)"$', r'\1', generated_text)

                    # prepare the JSON object with "text" and "label"
                    synth_sample = {
                        "text": generated_text,
                        "label": 0
                    }

                    # write the JSON object to the file
                    jsonl_file.write(json.dumps(synth_sample) + '\n')
                    pbar.update(1)
                    count += 1

                    if count >= default_config.num_keyword_groups_act:
                        print(f"Has reached {default_config.num_keyword_groups_act:} samples for {category}")
                        break

    print(f"Saved the synthetic samples to {output_file}")


def main(has_keywords=False):
    gpt_client = init_gpt()
    normal_label_list, _, original_task, _ = read_data_summary(default_config.data_name)
    
    normal_desc_dict = None
    if default_config.use_desc:
        normal_desc_dict = read_normal_desc(default_config.data_name, 
                                            "gpt", normal_label_list)
        
    if not has_keywords:
        generate_keywords_gpt(gpt_client, 
                              default_config.data_name, original_task, 
                              normal_label_list, normal_desc_dict,
                              default_config.num_keyword_groups)
    
    generate_sample_with_keywords_gpt(gpt_client, 
                                      default_config.data_name, original_task)


if __name__ == "__main__":
    # set has_keywords to True to skip the keyword generation if already generated
    main(has_keywords=False)
