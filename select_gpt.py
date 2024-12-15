import numpy as np
from utils import read_data_summary, TextDataset
from openai import OpenAI
from prompt.select_gpt_prompt import generate_model_selection_prompt
from config import DefaultConfig, PrivacyConfig

default_config = DefaultConfig()
privacy_config = PrivacyConfig()
# set display_flag to True to display the prompt for inspection
display_flag = False


def init_gpt():
    client = OpenAI(
        organization=privacy_config.organization,
        project=privacy_config.project,
        api_key=privacy_config.gpt_api_key
    )
    return client


def run_gpt(gpt_client, name, size, origianl_task,
            normal_label_list, anomaly_label_list,
            avg_len, max_len, min_len, std_len,
            normal_text, anomaly_text):
    prompt = generate_model_selection_prompt(name, size, origianl_task, 
                                             normal_label_list, anomaly_label_list,
                                             avg_len, max_len, min_len, std_len,
                                             normal_text, anomaly_text)
    global display_flag
    if display_flag:
        print("Here is the prompt for inspection:")
        print(prompt[0]["content"])
        display_flag = False

    response = gpt_client.chat.completions.create(
        # model=default_config.gpt_model_id,
        model="o1-preview",
        messages=prompt,
        max_completion_tokens=default_config.more_max_new_tokens,
        seed=default_config.seed,
        temperature=1
    )
    generated_text = response.choices[0].message.content
    print(generated_text)


def main():
    gpt_client = init_gpt()
    normal_label_list, anomaly_label_list, origianl_task, size = \
        read_data_summary(default_config.data_name)
    
    # compute average length of text, max length, min length, standard deviation
    dataset = TextDataset(default_config.data_name)
    X = dataset.get_X()
    len_array = np.array([len(x) for x in X])
    avg_len = np.mean(len_array)
    max_len = np.max(len_array)
    min_len = np.min(len_array)
    std_len = np.std(len_array)

    # iterate dataset to get the first normal text and anomaly text
    normal_text = None
    anomaly_text = None
    y = dataset.get_labels()
    for i in range(len(y)):
        if y[i] == 0:
            normal_text = X[i]
            break
    # reverse iterate the dataset
    for i in range(len(y)-1, -1, -1):
        if y[i] == 1:
            anomaly_text = X[i]
            break

    run_gpt(gpt_client, default_config.data_name, size, origianl_task, 
            normal_label_list, anomaly_label_list,
            avg_len, max_len, min_len, std_len,
            normal_text, anomaly_text)
    

if __name__ == "__main__":
    main()
