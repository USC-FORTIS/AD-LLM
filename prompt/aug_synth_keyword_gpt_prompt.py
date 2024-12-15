# GPT version
# the prompt for generating keyword groups in synthetic data augmentation task
def generate_keywords_prompt(name, original_task, normal_label_list, 
                             normal_desc_dict=None, num_keyword_groups=10):
    # whether to use the description
    if normal_desc_dict is None or len(normal_desc_dict) == 0:
        normal_categories = "\n".join([f'- {category}' for category in normal_label_list])
    else:
        normal_categories = "\n".join([f'- {category}:\n    - Description: {normal_desc_dict[category]}' for category in normal_label_list])
        
    system_message = f"""
You are an intelligent and professional assistant that generates groups of keywords for given categories in a dataset.
## Task:
- Following the rules below, generate **exactly**  {num_keyword_groups} unique keyword groups for **each given category** according to your understanding of the category (and its description).
- Each keyword group will be used to generate synthetic data for the corresponding category.

## Rules:
1. **Keyword Group Generation**:
    - For **each given category**, generate **exactly** {num_keyword_groups} keyword groups. Each group should contain exactly three keywords, with different levels of granularity: one broad/general, one intermediate, and one fine-grained.
    - Ensure that the three keywords in each group are thematically related to each other and align with the category's description.
    - Avoid redundancy or overly similar keywords across different groups.
    - Ensure that each group is unique and relevant to the key topics described in the category.
2. **Granularity**:
    - The first keyword should be broad/general, representing a high-level or overarching topic.
    - The second keyword should be intermediate, more specific than the first, but not overly narrow.
    - The third keyword should be fine-grained and specific, related to detailed subtopics or precise aspects of the category.
3. **Response Format**:
    - For each given category, provide the keyword groups as a list, where each entry is a group of three keywords (broad, intermediate, fine-grained).
    - Structure the response so that the key is the category name, and the value is a list of generated keyword groups.
    - Ensure the JSON output is properly formatted, including correct placement of commas between key-value pairs and no missing brackets.
    - Add a backslash (\) before any double quotation marks (") within the values of JSON output for proper parsing (i.e., from " to \\"), and ensure that single quotation marks (') are preserved without escaping.
"""
    user_message = f"""
The "{name}" dataset's original task is {original_task}. It contains the following category(ies):
{normal_categories}
"""
    assistant_message = f"Response in JSON format:\n"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]

    return messages
