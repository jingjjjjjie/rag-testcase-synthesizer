import re
import os
import random
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src import PROJECT_ROOT
from src.tools.json_utils import load_json,save_json
from src.tools.api import call_api_qwen
from src.tools.string_utils import read_text_file
from src.tools.rag_utils import expand_numbers_and_ranges

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))


def parse_transformations(text):
    # Regex pattern to match each transformation block
    pattern = re.compile(r'''
        ^(\d+)\.\s*<transformed-action>(.*?)</transformed-action>\s*            # Capture the transformation type
        <transformed-explanation>(.*?)</transformed-explanation>\s* # Capture the explanation
        <transformed-question>(.*?)</transformed-question>\s*                  # Capture the result
        <transformed-answer>(.*?)</transformed-answer>                  # Capture the answer
        ''', re.MULTILINE | re.DOTALL | re.VERBOSE)
    
    transformations = []
    
    # Find all matches in the text
    matches = pattern.findall(text)
    
    for match in matches:
        transformation = {
            'transformation': match[1].strip(),
            'explanation': match[2].strip(),
            'result': match[3].strip(),
            'answer': match[4].strip()
        }
        transformations.append(transformation)
    
    return transformations

class RephraseGeneratorPart:
    def __init__(self, save_interval=20):
        # token usage tracker
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.token_lock = threading.Lock()

        self.REPHRASE_GENERATOR_PART_INPUT_PATH, self.REPHRASE_GENERATOR_PART_PROMPT_PATH, self.REPHRASE_GENERATOR_PART_OUTPUT_PATH = None, None, None
        if os.getenv("REPHRASE_GENERATOR_PART_CONTENT_INPUT_PATH", None) != None:
            self.REPHRASE_GENERATOR_PART_INPUT_PATH = os.getenv("REPHRASE_GENERATOR_PART_CONTENT_INPUT_PATH")
            self.REPHRASE_GENERATOR_PART_PROMPT_PATH = os.getenv("REPHRASE_GENERATOR_PART_CONTENT_PROMPT_PATH")
            self.REPHRASE_GENERATOR_PART_OUTPUT_PATH = os.getenv("REPHRASE_GENERATOR_PART_CONTENT_OUTPUT_PATH")
            self.REPHRASE_GENERATOR_PART_GENERATED_TYPE = 'content'
        elif os.getenv("REPHRASE_GENERATOR_PART_ENTITYGRAPH_INPUT_PATH", None) != None:
            self.REPHRASE_GENERATOR_PART_INPUT_PATH = os.getenv("REPHRASE_GENERATOR_PART_ENTITYGRAPH_INPUT_PATH")
            self.REPHRASE_GENERATOR_PART_PROMPT_PATH = os.getenv("REPHRASE_GENERATOR_PART_ENTITYGRAPH_PROMPT_PATH")
            self.REPHRASE_GENERATOR_PART_OUTPUT_PATH = os.getenv("REPHRASE_GENERATOR_PART_ENTITYGRAPH_OUTPUT_PATH")
            self.REPHRASE_GENERATOR_PART_GENERATED_TYPE = 'entity_graph'
        else:
            raise EnvironmentError("Environment variable 'REPHRASE_GENERATOR_PART_CONTENT_INPUT_PATH' or 'REPHRASE_GENERATOR_PART_ENTITYGRAPH_INPUT_PATH' is not set.")
        
        self.REPHRASE_GENERATOR_PART_INPUT_PATH = os.path.join(PROJECT_ROOT, self.REPHRASE_GENERATOR_PART_INPUT_PATH)
        self.REPHRASE_GENERATOR_PART_PROMPT_PATH = os.path.join(PROJECT_ROOT, self.REPHRASE_GENERATOR_PART_PROMPT_PATH)
        self.REPHRASE_GENERATOR_PART_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.REPHRASE_GENERATOR_PART_OUTPUT_PATH)

        self.REPHRASE_GENERATOR_STOP_WORDS = os.getenv("REPHRASE_GENERATOR_STOP_WORDS", None)
        self.REPHRASE_GENERATOR_MAX_NEW_TOKENS = os.getenv("REPHRASE_GENERATOR_MAX_NEW_TOKENS", None)
        self.REPHRASE_GENERATOR_NUM_WORKERS = int(os.getenv("REPHRASE_GENERATOR_NUM_WORKERS", 4))
        self.REPHRASE_GENERATOR_MAX_GEN_TIMES = int(os.getenv("REPHRASE_GENERATOR_MAX_GEN_TIMES", 300))

        if os.path.exists(self.REPHRASE_GENERATOR_PART_OUTPUT_PATH):
            self.inputs = load_json(self.REPHRASE_GENERATOR_PART_OUTPUT_PATH)
            print(f"Loaded rephrase generator {len(self.inputs)} examples from {os.path.relpath(self.REPHRASE_GENERATOR_PART_OUTPUT_PATH, PROJECT_ROOT)}.")
        else:
            self.inputs = load_json(self.REPHRASE_GENERATOR_PART_INPUT_PATH)
            print(f"Loaded rephrase generator {len(self.inputs)} examples from {os.path.relpath(self.REPHRASE_GENERATOR_PART_INPUT_PATH, PROJECT_ROOT)}.")

        if self.REPHRASE_GENERATOR_MAX_GEN_TIMES == -1:
            self.REPHRASE_GENERATOR_MAX_GEN_TIMES = len(self.inputs)

        self.save_interval = save_interval

    def run(self):
        # load the generator prompt
        rephrase_generator_prompt = read_text_file(self.REPHRASE_GENERATOR_PART_PROMPT_PATH)
        print(f"Loaded rephrase generator prompt from {os.path.relpath(self.REPHRASE_GENERATOR_PART_PROMPT_PATH, PROJECT_ROOT)}.")

        all_num, success_num = 0, 0
        tasks = []

        with ThreadPoolExecutor(max_workers=self.REPHRASE_GENERATOR_NUM_WORKERS) as executor:
            if self.REPHRASE_GENERATOR_PART_GENERATED_TYPE in ['content']:
                for i, cur_input in enumerate(self.inputs[:self.REPHRASE_GENERATOR_MAX_GEN_TIMES]):

                    questions = cur_input['proposed-questions']
                    objective_facts = cur_input['objective-facts']

                    for proposed_question_type, proposed_question_dict in questions.items():
                        if 'rephrased-questions-part' in proposed_question_dict and proposed_question_dict['rephrased-questions-part']:
                            continue
                        needed_objective_fact_ids = proposed_question_dict['objective-facts']
                        needed_objective_fact_ids = re.findall(r'\d+-\d+|\d+', needed_objective_fact_ids)
                        needed_objective_fact_ids = expand_numbers_and_ranges(needed_objective_fact_ids)
                        needed_objective_factid_2_fact = {idx: objective_facts[idx-1] for idx in needed_objective_fact_ids if idx <= len(objective_facts)}

                        context = ""
                        # context = "Given clues:\n"
                        # for idx, clue in needed_objective_factid_2_fact.items():
                        #     context += f"{idx}. {clue}\n"
                        # context += "\n"
                        if 'positive' not in proposed_question_dict:
                            continue
                        context += f"Original Question: {proposed_question_dict['question']}\n"
                        context += f"Answer: {proposed_question_dict['positive']}\n"
                        context += "\n"

                        cur_rephrase_generator_prompt = rephrase_generator_prompt.replace('[[CONTEXT]]', context)
                        future = executor.submit(call_api_qwen, cur_rephrase_generator_prompt, TEMPERATURE)
                        tasks.append((future, proposed_question_dict))

            elif self.REPHRASE_GENERATOR_PART_GENERATED_TYPE in ['entity_graph']:
                for i, cur_input in list(self.inputs.items())[:self.REPHRASE_GENERATOR_MAX_GEN_TIMES]:

                    questions = cur_input['proposed-questions']
                    objective_relationship_prompts = cur_input['selected-relationships']['objective-relationship-prompts']

                    for proposed_question_type, proposed_question_dict in questions.items():
                        if 'rephrased-questions-part' in proposed_question_dict and proposed_question_dict['rephrased-questions-part']:
                            continue
                        needed_objective_relationship_ids = proposed_question_dict['objective-relationship-id']
                        needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', needed_objective_relationship_ids)
                        needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
                        needed_objective_relationship_id_2_prompt = {idx: objective_relationship_prompts[idx-1] for idx in needed_objective_relationship_ids if idx <= len(objective_relationship_prompts)}

                        context = ""
                        # context = "Given clues:\n"
                        # for idx, clue in needed_objective_relationship_id_2_prompt.items():
                        #     context += f"{idx}. {clue}\n"
                        # context += "\n"
                        if 'positive' not in proposed_question_dict:
                            continue
                        context += f"Original Question: {proposed_question_dict['question']}\n"
                        context += f"Answer: {proposed_question_dict['positive']}\n"
                        context += "\n"

                        cur_rephrase_generator_prompt = rephrase_generator_prompt.replace('[[CONTEXT]]', context)
                        future = executor.submit(call_api_qwen, cur_rephrase_generator_prompt, TEMPERATURE)
                        tasks.append((future, proposed_question_dict))
            else:
                raise ValueError(f"Invalid value for 'REPHRASE_GENERATOR_PART_GENERATED_TYPE': {self.REPHRASE_GENERATOR_PART_GENERATED_TYPE}")

            all_num = len(tasks)
            for future_info in tqdm(as_completed([t[0] for t in tasks]), total=len(tasks), desc="Generating", dynamic_ncols=True):
                future = future_info
                idx = [t[0] for t in tasks].index(future)
                if idx == -1:
                    raise ValueError("Invalid index.")
                proposed_question_dict = tasks[idx][1]
                try:
                    rephrase_generator_response, prompt_tokens, completion_tokens, _ = future.result(timeout=10*60)

                    # Thread-safe token accumulation
                    with self.token_lock:
                        self.total_prompt_tokens += prompt_tokens
                        self.total_completion_tokens += completion_tokens

                    rephrased_questions = parse_transformations(rephrase_generator_response)
                    if rephrased_questions:
                        proposed_question_dict['rephrased-questions-part'] = rephrased_questions  # inplace update
                        success_num += 1
                        if success_num % self.save_interval == 0:
                            print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.REPHRASE_GENERATOR_PART_OUTPUT_PATH, PROJECT_ROOT)}.')
                            save_json(self.inputs, self.REPHRASE_GENERATOR_PART_OUTPUT_PATH)
                except Exception as e:
                    print(f"Error processing question: {e}")

        if success_num or not os.path.exists(self.REPHRASE_GENERATOR_PART_OUTPUT_PATH):
            print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.REPHRASE_GENERATOR_PART_OUTPUT_PATH, PROJECT_ROOT)}.')
            save_json(self.inputs, self.REPHRASE_GENERATOR_PART_OUTPUT_PATH)

        print(f"Total prompt tokens: {self.total_prompt_tokens}")
        print(f"Total completion tokens: {self.total_completion_tokens}")
        print(f"Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%" if all_num > 0 else "Success rate: N/A")

        return self.total_prompt_tokens, self.total_completion_tokens, success_num, all_num