import os
import random
import json
import re
from tqdm import tqdm
import threading

from .. import PROJECT_ROOT
from src.components.entity_graph_constructor import EntityRelationshipGraph
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.tools.api import call_api_qwen
from src.tools.string_utils import reformat_objective_facts, convert_set_to_list 
from src.tools.string_utils import read_text_file
from src.tools.json_utils import save_json, load_json


TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

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

def extract_execution_output_entity_graph(text):
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
                r"<question>(.*?)</question>\s*<objective-relationship-id>(.*?)</objective-relationship-id>\s*"
                r"<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>",
                re.S
            )
            ret = [
                {
                    "question": match.group(1).strip(),
                    "objective-relationship-id": match.group(2).strip(),
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

def count_actual_numbers(numbers_and_ranges):
    total_count = 0
    for item in numbers_and_ranges:
        if '-' in item:  # It's a range like 'xx1-xx2'
            start, end = map(int, item.split('-'))
            total_count += (end - start + 1)
        else:  # It's a single number
            total_count += 1
    return total_count

class ProposeGenerator:
    def __init__(self, save_interval=10):
        self.PROPOSE_GENERATOR_INPUT_PATH, self.PROPOSE_GENERATOR_PROMPT_PATH, self.PROPOSE_GENERATOR_OUTPUT_PATH = None, None, None
        if os.getenv("PROPOSE_GENERATOR_CONTENT_INPUT_PATH", None) != None:
            self.PROPOSE_GENERATOR_INPUT_PATH = os.getenv("PROPOSE_GENERATOR_CONTENT_INPUT_PATH")
            self.PROPOSE_GENERATOR_PROMPT_PATH = os.getenv("PROPOSE_GENERATOR_CONTENT_PROMPT_PATH")
            self.PROPOSE_GENERATOR_OUTPUT_PATH = os.getenv("PROPOSE_GENERATOR_CONTENT_OUTPUT_PATH")
            self.PROPOSE_GENERATOR_GENERATED_TYPE = "content"
        elif os.getenv("PROPOSE_GENERATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
            self.PROPOSE_GENERATOR_INPUT_PATH = os.getenv("PROPOSE_GENERATOR_ENTITYGRAPH_INPUT_PATH")
            self.PROPOSE_GENERATOR_PROMPT_PATH = os.getenv("PROPOSE_GENERATOR_ENTITYGRAPH_PROMPT_PATH")
            self.PROPOSE_GENERATOR_OUTPUT_PATH = os.getenv("PROPOSE_GENERATOR_ENTITYGRAPH_OUTPUT_PATH")
            self.PROPOSE_GENERATOR_GENERATED_TYPE = "entity_graph"
        else:
            raise EnvironmentError("Environment variable 'PROPOSE_GENERATOR_CONTENT_INPUT_PATH' or 'PROPOSE_GENERATOR_ENTITYGRAPH_INPUT_PATH' is not set.")
        
        # token usage tracker
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.token_lock = threading.Lock()

        self.PROPOSE_GENERATOR_INPUT_PATH = os.path.join(PROJECT_ROOT, self.PROPOSE_GENERATOR_INPUT_PATH)
        self.PROPOSE_GENERATOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, self.PROPOSE_GENERATOR_PROMPT_PATH)
        self.PROPOSE_GENERATOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.PROPOSE_GENERATOR_OUTPUT_PATH)

        self.PROPOSE_GENERATOR_NUM_WORKERS = int(os.getenv("PROPOSE_GENERATOR_NUM_WORKERS", 4))
        self.PROPOSE_GENERATOR_MAX_GEN_TIMES = int(os.getenv("PROPOSE_GENERATOR_MAX_GEN_TIMES", 300))

        if self.PROPOSE_GENERATOR_GENERATED_TYPE in ['content']:
            if os.path.exists(self.PROPOSE_GENERATOR_OUTPUT_PATH):
                self.inputs = load_json(self.PROPOSE_GENERATOR_OUTPUT_PATH)
                print(f"Loaded propose generator {len(self.inputs)} examples from {os.path.relpath(self.PROPOSE_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.")
            else:
                self.inputs = load_json(self.PROPOSE_GENERATOR_INPUT_PATH)
                print(f"Loaded propose generator {len(self.inputs)} examples from {os.path.relpath(self.PROPOSE_GENERATOR_INPUT_PATH, PROJECT_ROOT)}.")
            
        elif self.PROPOSE_GENERATOR_GENERATED_TYPE in ['entity_graph']:
            self.inputs = load_json(self.PROPOSE_GENERATOR_INPUT_PATH)
            # convert the propose generator input into entity relationship graph
            self.entity_relationship_graph = EntityRelationshipGraph(self.inputs)
            print(f"Loaded propose generator {len(self.inputs)} examples from {os.path.relpath(self.PROPOSE_GENERATOR_INPUT_PATH, PROJECT_ROOT)}.")
        else:
            raise ValueError(f"Invalid value for 'PROPOSE_GENERATOR_GENERATED_TYPE': {self.PROPOSE_GENERATOR_GENERATED_TYPE}")

        if self.PROPOSE_GENERATOR_MAX_GEN_TIMES == -1:
            self.PROPOSE_GENERATOR_MAX_GEN_TIMES = len(self.inputs)

        self.save_interval = save_interval

    def _entity_relationship_graph_2_entity_relationship_prompt(self, entity_relationship_graph, strategy="random_relationship"):
        
        already_have_chunks = set()
        objective_relationships = []
        objective_relationship_prompts = []
        for relationship_item in entity_relationship_graph['relationships']:
            chunk_id = relationship_item['id']
            if chunk_id in already_have_chunks:
                continue
            already_have_chunks.add(chunk_id)
            objective_relationships.append(relationship_item)
            objective_relationship_prompts.append(f"<source_entity_name>{relationship_item['source_entity_name']}</source_entity_name>\n<target_entity_name>{relationship_item['target_entity_name']}</target_entity_name>\n<relationship_desc>{relationship_item['relationship_description']}</relationship_desc>")
        
        if len(objective_relationship_prompts) == 0:
            return "", [], []

        if strategy == "random_relationship":
            
            objective_idx_list = list(range(len(objective_relationship_prompts)))
            random_objective_idx_list = random.sample(objective_idx_list, min(10, len(objective_idx_list)))
            
            random_objective_relationships = [objective_relationships[idx] for idx in random_objective_idx_list]
            random_objective_relationship_prompts = [objective_relationship_prompts[idx] for idx in random_objective_idx_list]
            random_objective_relationship_prompts_with_numbers = [f"{idx+1}. {relationship_prompt}" for idx, relationship_prompt in enumerate(random_objective_relationship_prompts)]

            entity_relationship_prompt = "Objective Relationships:\n" + "\n".join(random_objective_relationship_prompts_with_numbers) + "\n\n"
            return entity_relationship_prompt, random_objective_relationship_prompts, random_objective_relationships
        else:
            raise NotImplementedError(f"Invalid value for 'strategy': {strategy}")

    def process_input_content(self, cur_input, cur_propose_generator_prompt, i):
        try:
            propose_generator_response, prompt_tokens, completion_tokens, _ = call_api_qwen(cur_propose_generator_prompt, TEMPERATURE)
            
            # Thread-safe token accumulation
            with self.token_lock:
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
            
            proposed_questions = extract_execution_output_content(propose_generator_response)
            result = {
                **cur_input,
                'proposed-questions': proposed_questions
            }
            return result, i
        except Exception as e:
            print(f"An error occurred while processing input {cur_input.get('id', 'unknown id')}: {e}")
            return None, None

    def run(self):

        # read the propose generator prompt
        # mult purposed, differenciated in the __init__ function, load from "content" path or "entity graph" path
        purpose_generator_prompt = read_text_file(self.PROPOSE_GENERATOR_PROMPT_PATH)
        print(f"Loaded propose generator prompt from {os.path.relpath(self.PROPOSE_GENERATOR_PROMPT_PATH, PROJECT_ROOT)}.")

        if self.PROPOSE_GENERATOR_GENERATED_TYPE in ['content']:

            success_num, all_num = 0, 0
            tasks = []
            
            with ThreadPoolExecutor(max_workers=self.PROPOSE_GENERATOR_NUM_WORKERS) as executor:
                for i, cur_input in enumerate(self.inputs[:self.PROPOSE_GENERATOR_MAX_GEN_TIMES]):
                    if 'proposed-questions' in cur_input:
                        continue
                    
                    context = reformat_objective_facts(cur_input)
                    cur_propose_generator_prompt = purpose_generator_prompt.replace('[[CONTEXT]]', context)
                    future = executor.submit(self.process_input_content, cur_input, cur_propose_generator_prompt, i)
                    tasks.append(future)

                all_num = len(tasks)
                for future in tqdm(as_completed(tasks), total=len(tasks), desc="Generating", dynamic_ncols=True):
                    try:
                        result, i = future.result(timeout=10*60)

                        self.inputs[i] = result
                        
                        success_num += 1
                        if success_num % self.save_interval == 0:
                            print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.PROPOSE_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.')
                            save_json(self.inputs,self.PROPOSE_GENERATOR_OUTPUT_PATH)
                    except Exception as e:
                        print(f"Error during processing: {e}")

            if success_num or not os.path.exists(self.PROPOSE_GENERATOR_OUTPUT_PATH):
                print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.PROPOSE_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.')
                save_json(self.inputs,self.PROPOSE_GENERATOR_OUTPUT_PATH)

            print(f"Total prompt tokens: {self.total_prompt_tokens}")
            print(f"Total completion tokens: {self.total_completion_tokens}")
            print(f"Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%" if all_num > 0 else "Success rate: N/A")

            return self.total_prompt_tokens, self.total_completion_tokens, success_num, all_num

        elif self.PROPOSE_GENERATOR_GENERATED_TYPE in ['entity_graph']:
            outputs = {}
            if os.path.exists(self.PROPOSE_GENERATOR_OUTPUT_PATH):
                outputs = load_json(self.PROPOSE_GENERATOR_OUTPUT_PATH)
                print(f"Loaded {len(outputs)} outputs from {os.path.relpath(self.PROPOSE_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.")

            already_done_entity_ids = set(outputs.keys())
            already_done_entity_ids = set([int(entity_id) for entity_id in already_done_entity_ids])

            all_num, success_num = 0, 0
            tasks = []

            with ThreadPoolExecutor(max_workers=self.PROPOSE_GENERATOR_NUM_WORKERS) as executor:
                for cur_entity_id, cur_entity_item in list(self.entity_relationship_graph.items())[:self.PROPOSE_GENERATOR_MAX_GEN_TIMES]:
                    if cur_entity_id in already_done_entity_ids:
                        continue
                    public_entity_name = cur_entity_item['entity_name']

                    subgraph_depth_1 = self.entity_relationship_graph.get_subgraph(cur_entity_id, depth=1)
                    subgraph_depth_1 = convert_set_to_list(subgraph_depth_1)
                    entity_relationship_prompt, \
                        cur_objective_relationship_prompts, \
                            cur_objective_relationships = self._entity_relationship_graph_2_entity_relationship_prompt(subgraph_depth_1)
                    
                    if entity_relationship_prompt == "" or len(cur_objective_relationship_prompts) <= 1:
                        continue
                    
                    cur_propose_generator_prompt = purpose_generator_prompt.replace('[[ENTITY_NAME]]', public_entity_name)
                    cur_propose_generator_prompt = cur_propose_generator_prompt.replace('[[CONTEXT]]', entity_relationship_prompt)
                    future = executor.submit(call_api_qwen, cur_propose_generator_prompt, TEMPERATURE)
                    tasks.append((future, cur_entity_id, subgraph_depth_1, cur_objective_relationships, cur_objective_relationship_prompts))

                all_num = len(tasks)
                for future in tqdm(as_completed([t[0] for t in tasks]), total=len(tasks), desc="Generating", dynamic_ncols=True):
                    idx = [t[0] for t in tasks].index(future)
                    if idx == -1:
                        raise ValueError("Invalid index.")
                    cur_entity_id, subgraph_depth_1, cur_objective_relationships, cur_objective_relationship_prompts = tasks[idx][1], tasks[idx][2], tasks[idx][3], tasks[idx][4]
                    
                    try:
                        propose_generator_response, prompt_tokens, completion_tokens, _ = future.result(timeout=10*60)

                        # Thread-safe token accumulation
                        with self.token_lock:
                            self.total_prompt_tokens += prompt_tokens
                            self.total_completion_tokens += completion_tokens

                        tmp_proposed_questions = extract_execution_output_entity_graph(propose_generator_response)
                        proposed_questions = {}
                        for tmp_proposed_question_type, tmp_proposed_question_dict in tmp_proposed_questions.items():
                            if "objective-relationship-id" in tmp_proposed_question_dict:
                                objective_relationship_ids = re.findall(r'\d+-\d+|\d+', tmp_proposed_question_dict["objective-relationship-id"])
                                actual_number_count = count_actual_numbers(objective_relationship_ids)
                                if actual_number_count > 1:
                                    proposed_questions[tmp_proposed_question_type] = tmp_proposed_question_dict
                                else:
                                    continue
                            else:
                                continue

                        outputs[cur_entity_id] = ({
                            'relationships': subgraph_depth_1['relationships'],
                            'selected-relationships': {
                                'objective-relationships': cur_objective_relationships,
                                'objective-relationship-prompts': cur_objective_relationship_prompts,
                            },
                            'proposed-questions': proposed_questions
                        })

                        success_num += 1
                        if success_num % self.save_interval == 0:
                            save_json(outputs, self.PROPOSE_GENERATOR_OUTPUT_PATH)
                    except Exception as e:
                        print(f"Error processing entity {cur_entity_id}: {e}")

            if success_num or not os.path.exists(self.PROPOSE_GENERATOR_OUTPUT_PATH):
                print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.PROPOSE_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.')
                save_json(outputs, self.PROPOSE_GENERATOR_OUTPUT_PATH)

            print(f"Total prompt tokens: {self.total_prompt_tokens}")
            print(f"Total completion tokens: {self.total_completion_tokens}")
            print(f"Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%" if all_num > 0 else "Success rate: N/A")

            return self.total_prompt_tokens, self.total_completion_tokens, success_num, all_num