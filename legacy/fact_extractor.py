import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.tools.api import call_api_qwen
from src.tools.json_utils import save_json, load_json
from src.tools.string_utils import read_text_file
from src import PROJECT_ROOT


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
    def __init__(self, verbose):
        self.verbose = verbose
        self.FACT_EXTRACTOR_LLM_TEMPERATURE = None
        self.FACT_EXTRACTOR_MAX_WORKERS = None
        self.FACT_EXTRACTOR_PROMPT_PATH = None
        self.FACT_EXTRACTOR_INPUT_PATH = None
        self.FACT_EXTRACTOR_OUTPUT_PATH = None

        if os.getenv("FACT_EXTRACTOR_PROMPT_PATH", None) != None:
            self.FACT_EXTRACTOR_LLM_TEMPERATURE = float(os.getenv("FACT_EXTRACTOR_LLM_TEMPERATURE", "0.6"))
            self.FACT_EXTRACTOR_MAX_WORKERS = int(os.getenv("FACT_EXTRACTOR_MAX_WORKERS", "4"))
            self.FACT_EXTRACTOR_PROMPT_PATH = os.getenv("FACT_EXTRACTOR_PROMPT_PATH")
            self.FACT_EXTRACTOR_INPUT_PATH = os.getenv("FACT_EXTRACTOR_INPUT_PATH")
            self.FACT_EXTRACTOR_OUTPUT_PATH = os.getenv("FACT_EXTRACTOR_OUTPUT_PATH")
        else:
            raise EnvironmentError("Environment variables are not defined correctly")

        self.FACT_EXTRACTOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, self.FACT_EXTRACTOR_PROMPT_PATH)
        self.FACT_EXTRACTOR_INPUT_PATH = os.path.join(PROJECT_ROOT, self.FACT_EXTRACTOR_INPUT_PATH)
        self.FACT_EXTRACTOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.FACT_EXTRACTOR_OUTPUT_PATH)

    def process_input(self, cur_input, fact_extractor_prompt, i):
        """Process a single input to extract objective facts"""
        try:
            context = cur_input['context']
            cur_fact_extractor_prompt = fact_extractor_prompt.replace('[[CONTEXT]]', context)
            fact_extractor_response, prompt_tokens, completion_tokens, _ = call_api_qwen(
                cur_fact_extractor_prompt,
                temperature=self.FACT_EXTRACTOR_LLM_TEMPERATURE
            )
            objective_facts, sens = extract_objective_facts(fact_extractor_response)

            result = {
                **cur_input,
                'objective-facts': objective_facts,
                'sens': sens
            }
            return result, i, prompt_tokens, completion_tokens
        except Exception as e:
            if self.verbose:
                print(f"An error occurred while processing input {cur_input.get('id', 'unknown id')}: {e}")
            return None, None, 0, 0

    def run(self):
        """Main fact extraction pipeline"""
        if self.verbose:
            print("Loading fact extraction prompt and input data......")

        # Load prompts and input json file
        fact_extractor_prompt = read_text_file(self.FACT_EXTRACTOR_PROMPT_PATH)
        chunked_questions = load_json(self.FACT_EXTRACTOR_INPUT_PATH)

        all_num, success_num = 0, 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Multiple workers extracting facts concurrently
        if self.verbose:
            print(f"Extracting facts with {self.FACT_EXTRACTOR_MAX_WORKERS} workers......")

        with ThreadPoolExecutor(max_workers=self.FACT_EXTRACTOR_MAX_WORKERS) as executor:
            futures = []
            for i, cur_input in enumerate(chunked_questions):
                if 'objective-facts' not in cur_input:
                    futures.append(executor.submit(self.process_input, cur_input, fact_extractor_prompt, i))

            all_num = len(futures)

            if self.verbose:
                iterator = tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True, desc="Extracting facts")
            else:
                iterator = as_completed(futures)

            for future in iterator:
                result, i, prompt_tokens, completion_tokens = future.result(timeout=10*60)
                if result != None:
                    chunked_questions[i] = result
                    success_num += 1
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens

        save_json(chunked_questions, self.FACT_EXTRACTOR_OUTPUT_PATH)

        if self.verbose:
            print(f"Completed: {success_num}/{all_num} inputs processed successfully")

        return total_prompt_tokens, total_completion_tokens, success_num/all_num


if __name__ == "__main__":
    pass
