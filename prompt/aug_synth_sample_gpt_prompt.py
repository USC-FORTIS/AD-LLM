# GPT version
# the prompt for generating a single synthetic sample with one keyword group in data augmentation task
def generate_sample_with_keywords_only_prompt(name, original_task, category, keyword_group):
    system_message = f"""
You are an intelligent and professional assistant that generates a synthetic text sample based on a group of 3 keywords with different levels of granularity.
## Task:
- Generate a synthetic text sample that incorporates the provided group of 3 keywords (broad, intermediate, and fine-grained) listed below.
- The generated sample should align with the meanings and themes suggested by the keywords provided.

## Rules:
1. **Sample Characteristics**:
    - Generate a synthetic text sample that naturally incorporates the three provided keywords (broad, intermediate, and fine-grained).
    - Ensure that the text sample is coherent and contextually relevant to the themes suggested by the keywords.
2. **Keyword Usage**:
    - The three keywords must appear naturally within the content.
    - Ensure that the broad keyword sets the overall context, the intermediate keyword refines the discussion, and the fine-grained keyword offers more detailed insight into a specific subtopic.
3. **Response Format**:
    - Provide the generated sample as a single string response representing the text sample.
    - Ensure the output is in a readable format.
    - Do not include any additional messages or commentary.
    - Add a backslash (\) before any double quotation marks (") within the values of JSON output for proper parsing (i.e., from " to \\"), and ensure that single quotation marks (') are preserved without escaping.
"""
    user_message = f"""
The "{name}" dataset's original task is {original_task}. The category is "{category}", and the group of keywords to use is:
- Broad: {keyword_group[0]}
- Intermediate: {keyword_group[1]}
- Fine-grained: {keyword_group[2]}
"""
    assistant_message = f"Response in plain text:\n"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]

    return messages