import sys
import os
import re

from ..tools.api import call_api
from ..tools.json_utils import save_json, load_json
from ..tools.string_utils import read_text_file
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed



def process_input(cur_input, fact_extractor_prompt, i):
        try:
            context = cur_input['context']
            cur_fact_extractor_prompt = fact_extractor_prompt.replace('[[CONTEXT]]', context)
            fact_extractor_response = call_api(cur_fact_extractor_prompt,temperature=temperature_fact_extractor)
            objective_facts, sens = extract_objective_facts(fact_extractor_response)

            result = {
                **cur_input,
                'objective-facts': objective_facts,
                'sens': sens
            }
            return result, i
        except Exception as e:
            print(f"An error occurred while processing input {cur_input.get('id', 'unknown id')}: {e}")
            return None, None  # or you can return an error result


def extract_objective_facts(text):
    """
    Extracts objective facts and their referenced sentence numbers.

    Parameters:
        text (str): The input text content.

    Returns:
        tuple: A tuple containing two lists.
            - objective_facts: A list of detailed descriptions of the objective facts.
            - sen_numbers: A list of sentence numbers as a formatted string corresponding to each objective fact.
    """
    # Regex pattern to match <detailed-desc> and <sentences-used> blocks
    pattern = r'<detailed-desc>(.*?)</detailed-desc>\s*<sentences-used>\[Sen\s*([^\]]+)\]</sentences-used>'
    
    # Use re.findall to extract all matches
    matches = re.findall(pattern, text, re.DOTALL)
    
    objective_facts = []
    sen_numbers = []

    for desc, sensors in matches:
        # Append detailed description to the objective_facts list
        objective_facts.append(desc.strip())
        
        # Extract all numbers using regex
        numbers = [int(num) for num in re.findall(r'\d+', sensors)]
        # Sort numbers to ensure the ranges are correctly identified
        numbers.sort()
        
        # Process the numbers to detect ranges
        formatted_sens = []
        i = 0
        while i < len(numbers):
            start = numbers[i]
            while i < len(numbers) - 1 and numbers[i] + 1 == numbers[i + 1]:
                i += 1
            end = numbers[i]
            if start == end:
                formatted_sens.append(f"{start}")
            else:
                formatted_sens.append(f"{start}-{end}")
            i += 1
        
        # Create the formatted string
        sen_string = f"{','.join(formatted_sens)}"
        sen_numbers.append(sen_string)
    
    return objective_facts, sen_numbers





if __name__ == '__main__':
    FACT_EXTRACTOR_LLM_TEMPERATURE = 0.6
    FACT_EXTRACTOR_MAX_WORKERS = 4
    FACT_EXTRACTOR_PROMPT_PATH = 'src/prompts/fact_extractor.txt'
    FACT_EXTRACTOR_INPUT_PATH = 'src/data/chunked_questions.json'
    FACT_EXTRACTOR_OUTPUT_PATH = 'src/data/fact_extracted.json'
    

    # load prompts and input json file
    fact_extractor_prompt = read_text_file(fact_extractor_prompt_path)
    chunked_questions = load_json(chunked_questions_json_path) 

    all_num, success_num = 0, 0
    # multiple workers extracting facts concurrently
    with ThreadPoolExecutor(max_workers) as executor:
        futures = []
        for i, cur_input in enumerate(chunked_questions):
            if 'objective-facts' not in cur_input:
                futures.append(executor.submit(process_input, cur_input, fact_extractor_prompt, i))

        all_num = len(futures)
        for future in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True):
            result, i = future.result(timeout=10*60)
            if result != None:
                chunked_questions[i] = result
                success_num += 1

    save_json(chunked_questions, save_output_fact_extracted_json_path)