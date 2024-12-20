import json
import logging
import os

from constants import GEMINI_MODELS
from tot.thought_node import ThoughtNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_reward_response(reward_response):
    # TODO: preprocess reponse
    ranking = [int(num.strip()) for num in reward_response.split(',')]
    return ranking

def preprocess_resource_allocation_response(resource_allocation_response):
    # TODO: preprocess reponse
    return int(resource_allocation_response)

def preprocess_reasoning_step_response(reasoning_step_response):
    # TODO: preprocess reponse
    return reasoning_step_response

def format_llm_input(model_id, prompt):
    if model_id == "meta-llama/Llama-3.1-8B-Instruct":
        return [
            {"role": "user", "content": prompt},
        ]
    
    if model_id in GEMINI_MODELS:
        return prompt
    
    logger.error(f"Unrecognized model ID: {model_id}")
    return prompt

def format_llm_output(model_id, output):
    if model_id == "meta-llama/Llama-3.1-8B-Instruct":
        return output[0]["generated_text"][-1]["content"]
    
    if model_id in GEMINI_MODELS:
        return output

    logger.error(f"Unrecognized model ID: {model_id}")
    return output

def get_sibling_thoughts_string(siblings: list, id_prefix=True):
    result = ""

    for node in siblings:
        prefix = f"{node.id}." if id_prefix else "-"
        result += f"{prefix} {node.thought}\n"

    return result

def get_reasoning_path_string(reasoning_path: list):
    result = ""

    for r_step in reasoning_path:
        result += f"- {r_step}\n"

    return result

def preprocess_text_for_visualization(text: str):
    if not text:
        return "N/A"
    
    # Replace common LaTeX commands with Unicode equivalents
    latex_to_unicode = {
        r"\ge": "≥",
        r"\le": "≤",
        r"\ne": "≠"
    }
    
    for latex, unicode_char in latex_to_unicode.items():
        text = text.replace(latex, unicode_char)\
                   .replace("$", r"\$")

    # Replace other problematic characters
    # text = text.replace("\\", "\\\\")

    return text

def is_non_empty_file(file_path):
    try:
        return os.path.isfile(file_path) and os.path.getsize(file_path) > 0
    except OSError:
        return False

def save_input_output_for_finetuning(input: str, output: str, ft_dataset_path: str):
    mode = 'a' if is_non_empty_file(ft_dataset_path) else 'w'
    with open(ft_dataset_path, mode) as f:
        item = {
            "text_input": input,
            "output": output
        }
        json.dump(item, f)
        f.write('\n')

def save_branch_for_finetuning(thought_nodes: list[ThoughtNode], ft_dataset_path: str):
    finetuning_data = []

    for node in thought_nodes:
        current_node = node

        while current_node.tree_level > 0:
            prompt_response_pair = {
                "text_input": current_node.generation_prompt,
                "output": current_node.thought
            }
            finetuning_data.append(prompt_response_pair)
            current_node = current_node.parent
    
    mode = 'a' if is_non_empty_file(ft_dataset_path) else 'w'
    with open(ft_dataset_path, mode) as f:
        for item in finetuning_data:
            json.dump(item, f)
            f.write('\n')

def load_jsonl_data(file_path: str) -> list:
    data_list = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found at path: {file_path}")
        return data_list

    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data_item = json.loads(line)
                    data_list.append(data_item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()} - Error: {e}")
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")

    return data_list