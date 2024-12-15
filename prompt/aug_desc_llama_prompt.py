# Llama 3 and later version
# the prompt for generating category descriptions in data augmentation task
def generate_prompt(name, normal_label_list, anomaly_label_list, origianl_task):
    normal_categories = "\n".join([f'- {category}' for category in normal_label_list])
    anomaly_categories = "\n".join([f'- {category}' for category in anomaly_label_list])
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an intelligent and professional assistant that generates descriptions for given categories in a text dataset. 
## Task:
- Following the rules below, generate detailed textual descriptions that explain the main characteristics, typical topics, and common examples for each given category.

## Rules:
1. For each category, provide a continuous, coherent description in a single paragraph that includes:
    - **Definition or overview**: Start by briefly defining or describing the category in one to two sentences. If you list multiple aspects or features in the definition (such as related fields or industries), ensure you append expressions like "etc." or "and so on" to indicate that the list is not exhaustive.
    - **Main topics or subjects**: Highlight the typical topics or subjects covered by this category. Ensure that you use phrases like "etc." or "and so on" at the end of each list to indicate that the list is not exhaustive.
	- **Relevant examples**: Mention examples of content that belong to this category. Also, use expressions like "etc." or "and so on" at the end of the list to show that these are illustrative, not exhaustive.
2. Use **step-by-step reasoning** to ensure the descriptions are logical and clear.
3. Each description should be clear, coherent, and helpful for someone unfamiliar with the dataset and the task.
4. Always append phrases like "etc." or "and so on" to lists or enumerations of examples, topics, or aspects, **including the definition part**.
5. Response Format:
    - Provide a response where each key is the category name, and the value is the corresponding description as a continuous paragraph.
    - Ensure the JSON output is correctly formatted, including correct placement of commas between key-value pairs.
    - Add a backslash (\) before any double quotation marks (") within the values of JSON output for proper parsing (i.e., from " to \\"), and ensure that single quotation marks (') are preserved without escaping.
<|eot_id|><|start_header_id|>user<|end_header_id|>
The "{name}" dataset's original task is {origianl_task}. It contains the following categories:
{normal_categories}
{anomaly_categories}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Response in JSON format:
"""

    return prompt
