import re
import os
import copy
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from .. import PROJECT_ROOT

from ..tools.api import call_api_qwen
from ..tools.json_utils import load_json,save_json
from ..tools.string_utils import read_text_file

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))


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

class FactExtractor:
    def __init__(self, save_interval=10):
        self.FACT_EXTRACTOR_INPUT_PATH = None
        self.FACT_EXTRACTOR_PROMPT_PATH = None
        self.FACT_EXTRACTOR_OUTPUT_PATH = None

        if os.getenv("FACT_EXTRACTOR_INPUT_PATH", None) != None:
            self.FACT_EXTRACTOR_INPUT_PATH = os.getenv("FACT_EXTRACTOR_INPUT_PATH")
            self.FACT_EXTRACTOR_PROMPT_PATH = os.getenv("FACT_EXTRACTOR_PROMPT_PATH")
            self.FACT_EXTRACTOR_OUTPUT_PATH = os.getenv("FACT_EXTRACTOR_OUTPUT_PATH")
        else:
            raise EnvironmentError("Environment variable 'FACT_EXTRACTOR_INPUT_PATH' is not set.")

        self.FACT_EXTRACTOR_INPUT_PATH = os.path.join(PROJECT_ROOT, self.FACT_EXTRACTOR_INPUT_PATH)
        self.FACT_EXTRACTOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, self.FACT_EXTRACTOR_PROMPT_PATH)
        self.FACT_EXTRACTOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.FACT_EXTRACTOR_OUTPUT_PATH)

        self.FACT_EXTRACTOR_NUM_WORKERS = int(os.getenv("FACT_EXTRACTOR_NUM_WORKERS", 4))
        self.save_interval = save_interval

        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.token_lock = threading.Lock()
      

    def process_input(self, cur_input, fact_extractor_prompt, i):
        try:
            context = cur_input['context']
            cur_fact_extractor_prompt = fact_extractor_prompt.replace('[[CONTEXT]]', context)
            fact_extractor_response, prompt_tokens, completion_tokens, _ = call_api_qwen(cur_fact_extractor_prompt, TEMPERATURE)

            # Thread-safe token accumulation
            with self.token_lock:
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens

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

    def run(self):
        if os.path.exists(self.FACT_EXTRACTOR_OUTPUT_PATH):
            inputs = load_json(self.FACT_EXTRACTOR_OUTPUT_PATH)
            print(f"Loaded {len(inputs)} fact extractor examples from {os.path.relpath(self.FACT_EXTRACTOR_OUTPUT_PATH, PROJECT_ROOT)}.")
        else:
            inputs = load_json(self.FACT_EXTRACTOR_INPUT_PATH)
            print(f"Loaded {len(inputs)} fact extractor examples from {os.path.relpath(self.FACT_EXTRACTOR_INPUT_PATH, PROJECT_ROOT)}.")
        
        base_dir = os.path.dirname(self.FACT_EXTRACTOR_OUTPUT_PATH)
        os.makedirs(base_dir, exist_ok=True)
        chunk_output_path = os.path.join(base_dir, os.path.basename(self.FACT_EXTRACTOR_INPUT_PATH))
        if not os.path.exists(chunk_output_path):
            os.system(f"cp {self.FACT_EXTRACTOR_INPUT_PATH} {chunk_output_path}")
            print(f"Copied {os.path.relpath(self.FACT_EXTRACTOR_INPUT_PATH, PROJECT_ROOT)} to {os.path.relpath(chunk_output_path, PROJECT_ROOT)}.")

        fact_extractor_prompt = read_text_file(self.FACT_EXTRACTOR_PROMPT_PATH)  
        print(f"Loaded fact extractor prompt from {os.path.relpath(self.FACT_EXTRACTOR_PROMPT_PATH, PROJECT_ROOT)}.")

        all_num, success_num = 0, 0
        with ThreadPoolExecutor(max_workers=self.FACT_EXTRACTOR_NUM_WORKERS) as executor:
            futures = []
            for i, cur_input in enumerate(inputs):
                if 'objective-facts' not in cur_input:
                    futures.append(executor.submit(self.process_input, cur_input, fact_extractor_prompt, i))

            all_num = len(futures)
            for future in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True):
                result, i = future.result(timeout=10*60)
                if result != None:
                    inputs[i] = result
                    success_num += 1
                    if success_num % self.save_interval == 0:
                        print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.FACT_EXTRACTOR_OUTPUT_PATH, PROJECT_ROOT)}.')
                        save_json(inputs, self.FACT_EXTRACTOR_OUTPUT_PATH)
                        
        if success_num or not os.path.exists(self.FACT_EXTRACTOR_OUTPUT_PATH):
            print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.FACT_EXTRACTOR_OUTPUT_PATH, PROJECT_ROOT)}.')
            save_json(inputs, self.FACT_EXTRACTOR_OUTPUT_PATH)

        print(f"Total prompt tokens: {self.total_prompt_tokens}")
        print(f"Total completion tokens: {self.total_completion_tokens}")
        print(f"Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%" if all_num > 0 else "Success rate: N/A")

        return self.total_prompt_tokens, self.total_completion_tokens, success_num, all_num