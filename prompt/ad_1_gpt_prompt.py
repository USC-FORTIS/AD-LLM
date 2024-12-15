# Llama 3 and later version
# the prompt for the anomaly detection task with "Normal Only" setting
# enable the data augmentation of using LLM-generated category descriptions
def generate_prompt_setting_1(text, normal_label_list, anomaly_label_list=None, 
                              normal_desc_dict=None, anomaly_desc_dict=None, origianl_task=None):
    if normal_desc_dict is None or len(normal_desc_dict) == 0:
        normal_categories = "\n".join([f'- **{category}**' for category in normal_label_list])
        changable_part = 'each category listed in the "Categories" section above, considering factors such as main topics, meanings, background, sentiments, etc'
    else:
        # save spot for few-shot detection with examples
        normal_categories = "\n".join([f'- **{category}**:\n    - Description: {normal_desc_dict[category]}' for category in normal_label_list])
        changable_part = 'the description of each category listed in the "Categories" section above'
    
    system_message = f"""
You are an intelligent and professional assistant that detects anomalies in text data.
## Task:
- Following the rules below, determine whether the given text sample is an anomaly. Provide a brief explanation of your reasoning and assign an anomaly confidence score between 0 and 1.

## Category(ies):
{normal_categories}

## Rules:
1. **Anomaly Definition**:
    - A text sample is considered an **anomaly** if it does **not** belong to **any of the categories** listed above.
2. **Scoring**:
    - Assign an anomaly confidence score between 0 and 1.
    - Use higher scores when you are highly confident in your decision.
    - Use lower scores when you are uncertain or think the text sample is **not** an anomaly.
3. **Step-by-step Reasoning** (Chain of Thought):
    - **Step 1**. Read the entire text sample carefully and understand it thoroughly.
    - **Step 2**. Analyze the text sample by comparing its content to {changable_part}.
    - **Step 3**. Determine which category the text sample **most closely aligns with**.
        - If it aligns with any category, it is **not** an anomaly.
        - If it does **not** align with any category, it is an anomaly.
    - **Step 4**. Assign an anomaly confidence score based on how confident you are that the text sample is an anomaly.
4. **Additional Notes**:
    - A text sample may relate to multiple categories, but it should be classified into the **most relevant** one in this task.
    - If you are uncertain whether the text sample **significantly aligns** with a category, assume that it does, which means it is **not** an anomaly.
5. **Response Format**:
    - Provide responses in a strict **JSON** format with the keys "reason" and "anomaly_score."
        - "reason": Your brief explanation of the reasoning in one to three sentences logically.
        - "anomaly_score": Your anomaly confidence score between 0 and 1.
    - Ensure the JSON output is correctly formatted, including correct placement of commas between key-value pairs.
    - Add a backslash (\) before any double quotation marks (") within the values of JSON output for proper parsing (i.e., from " to \\"), and ensure that single quotation marks (') are preserved without escaping.
"""
    user_message = f"Text sample:\n" + '"' + text + '"'

    assistant_message = f"Response in JSON format:\n"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]

    return messages
