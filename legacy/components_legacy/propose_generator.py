import sys
import os
import re 
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(".."))
from tools.string_utils import read_text_file
from tools.json_utils import load_json
from tools.api import call_api

def extract_execution_output_content(text):
    """
    Extracts the structured content of "Execution" output dynamically for any question categories.
    Args:
        text (str): Input text containing "Execution" output.
    Returns:
        dict: A dictionary containing structured data for each dynamically matched question category.
    """
    # Define regex to capture all question categories and their content
    category_pattern = re.compile(r"\d+\.\s*<([^>]+-questions)>(.*?)((?=\d+\.\s*<[^>]+-questions>)|$)", re.S)

    # Parse content for each category dynamically
    categories = {}
    for match in category_pattern.finditer(text):
        category_name = match.group(1).strip()
        category_content = match.group(2).strip()

        # Function to parse individual questions within a category
        def parse_questions(content):
            question_pattern = re.compile(
                r"<question>(.*?)</question>\s*<objective-facts>(.*?)</objective-facts>\s*<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>",
                re.S
            )
            ret = [
                {
                    "question": match.group(1).strip(),
                    "objective-facts": match.group(2).strip(),
                    "reasoning": match.group(3).strip(),
                    "answer": match.group(4).strip()
                }
                for match in question_pattern.finditer(content)
            ]
            if len(ret) == 0:
                return []
            return ret

        # Parse and store questions for the current category
        ret = parse_questions(category_content)
        if len(ret) == 0:
            continue
        categories[category_name] = ret[0]

    return categories

def process_input_content(cur_input, cur_propose_generator_prompt, i):
        try:
            propose_generator_response = call_api(cur_propose_generator_prompt, temperature=0.6)
            proposed_questions = extract_execution_output_content(propose_generator_response)
            result = {
                **cur_input,
                'proposed-questions': proposed_questions
            }
            return result, i
        except Exception as e:
            print(f"An error occurred while processing input {cur_input.get('id', 'unknown id')}: {e}")
            return None, None 
        
def reformat_objective_facts(data):
    result = {"Objective Facts": []}

    # Reformat Objective Facts
    for idx, fact in enumerate(data['objective-facts'], start=1):
        result["Objective Facts"].append(
            f"{idx}. <detailed-desc>{fact}</detailed-desc>"
        )
    
    result_str = ""
    for key, values in result.items():
        result_str += f"{key}:\n" + "\n".join(values) + "\n"
    
    return result_str

inputs = load_json("../data/fact_extracted.json")
propose_generator_prompt = read_text_file("../prompts/propose_generator_content.txt")

PROPOSE_GENERATOR_NUM_WORKERS = 4
PROPOSE_GENERATOR_MAX_GEN_TIMES = 100
PROPOSE_GENERATOR_OUTPUT_PATH = '../data/proposed_questions.json'
save_interval = 10


def run():
    tasks = []
    success_num, all_num = 0, 0
    
    # create a pool of threads (parallel worker)
    with ThreadPoolExecutor(max_workers=PROPOSE_GENERATOR_NUM_WORKERS) as executor: 
        for i, cur_input in enumerate(inputs[:PROPOSE_GENERATOR_MAX_GEN_TIMES]): # only process questions (input) up to n
                if 'proposed-questions' in cur_input: # check if proposed questions already exist
                    continue #  skip inputs that contain 'proposed questions' key
                
                context = reformat_objective_facts(cur_input) 
                cur_propose_generator_prompt = propose_generator_prompt.replace('[[CONTEXT]]', context)
                #  Submit a background task to the thread pool
                future = executor.submit(process_input_content, cur_input, cur_propose_generator_prompt, i) # the original input, the prompt(with context inserted) index
                tasks.append(future) # stores each future so we can later wait for results

        all_num = len(tasks)
        # wait for tasks to finish 
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Generating", dynamic_ncols=True):
            try:
                result, i = future.result(timeout=10*60) # retrieves the result from the finished task, waits up to 10 minutes for each task
                inputs[i] = result # writes the result back into the original input list
                
                success_num += 1
                if success_num % save_interval == 0: # saves every "save_interval" successful inputs
                    dir_path = os.path.dirname(PROPOSE_GENERATOR_OUTPUT_PATH) # create output dir if needed
                    os.makedirs(dir_path, exist_ok=True)
                    print(f'Saving {success_num}/{all_num} outputs to {PROPOSE_GENERATOR_OUTPUT_PATH}.')
                    with open(PROPOSE_GENERATOR_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                        json.dump(inputs, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error processing input id {cur_input['id']}: {e}")
    if success_num or not os.path.exists(PROPOSE_GENERATOR_OUTPUT_PATH):
        dir_path = os.path.dirname(PROPOSE_GENERATOR_OUTPUT_PATH)
        os.makedirs(dir_path, exist_ok=True)
        print(f'Saving {success_num}/{all_num} outputs to {PROPOSE_GENERATOR_OUTPUT_PATH}.')
        with open(PROPOSE_GENERATOR_OUTPUT_PATH, 'w', encoding="utf-8") as f:
            json.dump(inputs, f, indent=2, ensure_ascii=False)

    return success_num, all_num