#You can't manually write scoring criteria for thousands of automatically generated questions. So this component automatically generates the evaluation rubric by asking an LLM: "Given this original question, this transformation, and this answer - what should we check to see if a RAG system answered it correctly?" In short: It creates the "answer key" for grading RAG systems on transformed questions.

import os
import json
import argparse
import threading
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import re

from src.tools.api import call_api_qwen
from src.tools.rag_utils import expand_numbers_and_ranges
from src.tools.json_utils import save_json, load_json
from .. import PROJECT_ROOT


class RephraseEvaluator:
    """
    A class to generate scoring criteria for rephrased questions.
    """

    def __init__(self, save_interval=100):
        """
        Initialize the RephraseEvaluator with configuration from environment variables.

        Args:
            save_interval: The interval at which to save intermediate results
        """
        self.PROJECT_ROOT = PROJECT_ROOT
        self.temperature = float(os.getenv("TEMPERATURE", 0.6))
        self.save_interval = save_interval

        # Load paths from environment
        self.REPHRASE_EVALUATOR_INPUT_PATH, self.REPHRASE_EVALUATOR_OUTPUT_PATH = None, None
        if os.getenv("REPHRASE_EVALUATOR_CONTENT_INPUT_PATH", None) != None:
            self.REPHRASE_EVALUATOR_INPUT_PATH = os.getenv("REPHRASE_EVALUATOR_CONTENT_INPUT_PATH")
            self.REPHRASE_EVALUATOR_OUTPUT_PATH = os.getenv("REPHRASE_EVALUATOR_CONTENT_OUTPUT_PATH")
        elif os.getenv("REPHRASE_EVALUATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
            self.REPHRASE_EVALUATOR_INPUT_PATH = os.getenv("REPHRASE_EVALUATOR_ENTITYGRAPH_INPUT_PATH")
            self.REPHRASE_EVALUATOR_OUTPUT_PATH = os.getenv("REPHRASE_EVALUATOR_ENTITYGRAPH_OUTPUT_PATH")
        else:
            raise EnvironmentError("Environment variable 'REPHRASE_EVALUATOR_CONTENT_INPUT_PATH' or 'REPHRASE_EVALUATOR_ENTITYGRAPH_INPUT_PATH' is not set.")

        self.REPHRASE_EVALUATOR_INPUT_PATH = os.path.join(PROJECT_ROOT, self.REPHRASE_EVALUATOR_INPUT_PATH)
        self.REPHRASE_EVALUATOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.REPHRASE_EVALUATOR_OUTPUT_PATH)
        self.REPHRASE_EVALUATOR_NUM_WORKERS = int(os.getenv("REPHRASE_EVALUATOR_NUM_WORKERS", 4))
        self.REPHRASE_EVALUATOR_MAX_GEN_TIMES = int(os.getenv("REPHRASE_EVALUATOR_MAX_GEN_TIMES", 300))

        # Token usage tracker
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.token_lock = threading.Lock()

        # Path to prompts directory
        self.prompts_dir = os.path.join(PROJECT_ROOT, "src", "prompts")

    def load_prompt(self, prompt_type: str) -> str:
        """Load prompt from file."""
        prompt_file = os.path.join(self.prompts_dir, f"{prompt_type}_evaluator.txt")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()

    def scoring_by_points_once_give_one(self, transformed_question, rephrased_type: str):
        """
        Score a transformed question by generating evaluation criteria.

        Args:
            transformed_question: The formatted question with transformation details
            rephrased_type: Type of rephrasing ('rephrased', 'rephrased_part', etc.)

        Returns:
            Tuple of (response, prompt_tokens, completion_tokens)
        """
        if rephrased_type == 'rephrased':
            cur_prompt = self.load_prompt('rephrased')
        elif rephrased_type == 'rephrased_part':
            cur_prompt = self.load_prompt('rephrased_part')
        elif rephrased_type == 'rephrased_hybrid':
            cur_prompt = self.load_prompt('rephrased_hybrid')
        elif rephrased_type == 'rephrased_hybrid_part':
            cur_prompt = self.load_prompt('rephrased_hybrid_part')
        else:
            raise ValueError("Invalid rephrased_type")

        cur_prompt = cur_prompt.replace("[[TRANSFORMED QUESTION]]", transformed_question)
        response, prompt_tokens, completion_tokens, _ = call_api_qwen(cur_prompt, self.temperature)
        return response, prompt_tokens, completion_tokens

    def get_transformed_question(self, last_question, last_answer, cur_rephrased_question):
        """Format a transformed question for evaluation."""
        ret_str = ""
        ret_str += f"Original Question: {last_question}\n"
        ret_str += f"Answer: {last_answer}\n\n"
        ret_str += "Rephrased Question:\n"
        ret_str += f"    <transformed-action>{cur_rephrased_question['transformation']}</transformed-action>\n"
        ret_str += f"    <transformed-explanation>{cur_rephrased_question['explanation']}</transformed-explanation>\n"
        ret_str += f"    <transformed-question>{cur_rephrased_question['result']}</transformed-question>\n"
        ret_str += f"    <transformed-answer>{cur_rephrased_question['answer']}</transformed-answer>\n"
        ret_str += "Scoring Criteria:\n"
        return ret_str

    def process_file_content(self):
        """Process content-based files for rephrase evaluation."""
        os.makedirs(os.path.dirname(self.REPHRASE_EVALUATOR_OUTPUT_PATH), exist_ok=True)

        if os.path.exists(self.REPHRASE_EVALUATOR_OUTPUT_PATH):
            data = load_json(self.REPHRASE_EVALUATOR_OUTPUT_PATH)
        else:
            data = load_json(self.REPHRASE_EVALUATOR_INPUT_PATH)

        rephrase_evaluator_max_gen_times = self.REPHRASE_EVALUATOR_MAX_GEN_TIMES
        if rephrase_evaluator_max_gen_times == -1:
            rephrase_evaluator_max_gen_times = len(data)

        all_num, new_gen_num = 0, 0

        with ThreadPoolExecutor(max_workers=self.REPHRASE_EVALUATOR_NUM_WORKERS) as executor:
            futures_to_data = {}
            for cur_dict in data[:rephrase_evaluator_max_gen_times]:

                if 'proposed-questions' not in cur_dict:
                    continue
                proposed_questions = cur_dict['proposed-questions']

                if_already_generated = False
                for proposed_question_type, proposed_question_dict in proposed_questions.items():

                    question = proposed_question_dict['question']
                    answer = proposed_question_dict['answer']

                    cur_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                    last_question = question
                    last_answer = answer
                    for cur_rephrased_question_part in cur_rephrased_questions_part:
                        if 'scoring-criteria' in cur_rephrased_question_part:
                            continue
                        transformed_question = self.get_transformed_question(last_question, last_answer, cur_rephrased_question_part)
                        future = executor.submit(self.scoring_by_points_once_give_one, transformed_question, rephrased_type='rephrased_part')
                        futures_to_data[future] = (cur_rephrased_question_part, 'criteria')
                        last_question = cur_rephrased_question_part['result']
                        last_answer = cur_rephrased_question_part['answer']

                    cur_rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                    last_question = question
                    last_answer = answer
                    for cur_rephrased_question_hybrid in cur_rephrased_questions_hybrid:
                        if 'scoring-criteria' in cur_rephrased_question_hybrid:
                            continue
                        transformation_type = cur_rephrased_question_hybrid['transformation']
                        if "Partial Transformation" in transformation_type:
                            transformed_question = self.get_transformed_question(last_question, last_answer, cur_rephrased_question_hybrid)
                            future = executor.submit(self.scoring_by_points_once_give_one, transformed_question, rephrased_type='rephrased_hybrid_part')
                            futures_to_data[future] = (cur_rephrased_question_hybrid, 'criteria')
                        last_question = cur_rephrased_question_hybrid['result']
                        last_answer = cur_rephrased_question_hybrid['answer']

                if if_already_generated:
                    continue

            all_num = len(futures_to_data)
            for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
                cur_rephrased_question, score_type = futures_to_data[future]
                try:
                    score_response, prompt_tokens, completion_tokens = future.result(timeout=5*60)

                    # Thread-safe token accumulation
                    with self.token_lock:
                        self.total_prompt_tokens += prompt_tokens
                        self.total_completion_tokens += completion_tokens

                    if score_type == 'criteria':
                        score_response = score_response.replace("Scoring Criteria:", "")
                        cur_rephrased_question['scoring-criteria'] = score_response
                    else:
                        raise ValueError("Invalid score_type")

                    new_gen_num += 1
                    if (new_gen_num + 1) % self.save_interval == 0:
                        print(f"Saving results to {os.path.relpath(self.REPHRASE_EVALUATOR_OUTPUT_PATH, PROJECT_ROOT)}")
                        save_json(data, self.REPHRASE_EVALUATOR_OUTPUT_PATH)
                        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

                except Exception as e:
                    print(f"Error processing criteria: {e}")
                    continue

        if new_gen_num or not os.path.exists(self.REPHRASE_EVALUATOR_OUTPUT_PATH):
            print(f"Saving results to {os.path.relpath(self.REPHRASE_EVALUATOR_OUTPUT_PATH, PROJECT_ROOT)}")
            save_json(data, self.REPHRASE_EVALUATOR_OUTPUT_PATH)
            print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

        print(f"Token Usage - Prompt: {self.total_prompt_tokens}, Completion: {self.total_completion_tokens}, Total: {self.total_prompt_tokens + self.total_completion_tokens}")
        return new_gen_num, all_num, self.total_prompt_tokens, self.total_completion_tokens

    def process_file_entity_graph(self):
        """Process entity graph-based files for rephrase evaluation."""
        os.makedirs(os.path.dirname(self.REPHRASE_EVALUATOR_OUTPUT_PATH), exist_ok=True)

        if os.path.exists(self.REPHRASE_EVALUATOR_OUTPUT_PATH):
            data = load_json(self.REPHRASE_EVALUATOR_OUTPUT_PATH)
        else:
            data = load_json(self.REPHRASE_EVALUATOR_INPUT_PATH)

        rephrase_evaluator_max_gen_times = self.REPHRASE_EVALUATOR_MAX_GEN_TIMES
        if rephrase_evaluator_max_gen_times == -1:
            rephrase_evaluator_max_gen_times = len(data)

        all_num, new_gen_num = 0, 0

        with ThreadPoolExecutor(max_workers=self.REPHRASE_EVALUATOR_NUM_WORKERS) as executor:
            futures_to_data = {}

            for entity_id, entity_dict in list(data.items())[:rephrase_evaluator_max_gen_times]:

                proposed_questions = entity_dict['proposed-questions']

                if_already_generated = False
                for proposed_question_type, proposed_question_dict in proposed_questions.items():
                    question = proposed_question_dict['question']
                    answer = proposed_question_dict['positive']

                    cur_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                    last_question = question
                    last_answer = answer
                    for cur_rephrased_question_part in cur_rephrased_questions_part:
                        if 'scoring-criteria' in cur_rephrased_question_part:
                            continue
                        transformed_question = self.get_transformed_question(last_question, last_answer, cur_rephrased_question_part)
                        future = executor.submit(self.scoring_by_points_once_give_one, transformed_question, rephrased_type='rephrased_part')
                        futures_to_data[future] = (cur_rephrased_question_part, 'criteria')
                        last_question = cur_rephrased_question_part['result']
                        last_answer = cur_rephrased_question_part['answer']

                    cur_rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                    last_question = question
                    last_answer = answer
                    for cur_rephrased_question_hybrid in cur_rephrased_questions_hybrid:
                        if 'scoring-criteria' in cur_rephrased_question_hybrid:
                            continue
                        transformed_question = self.get_transformed_question(last_question, last_answer, cur_rephrased_question_hybrid)
                        future = executor.submit(self.scoring_by_points_once_give_one, transformed_question, rephrased_type='rephrased_hybrid_part')
                        futures_to_data[future] = (cur_rephrased_question_hybrid, 'criteria')
                        last_question = cur_rephrased_question_hybrid['result']
                        last_answer = cur_rephrased_question_hybrid['answer']

                if if_already_generated:
                    continue

            all_num = len(futures_to_data)
            for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
                proposed_question_dict, score_type = futures_to_data[future]
                try:
                    cur_rephrased_question, prompt_tokens, completion_tokens = future.result(timeout=10*60)

                    # Thread-safe token accumulation
                    with self.token_lock:
                        self.total_prompt_tokens += prompt_tokens
                        self.total_completion_tokens += completion_tokens

                    if score_type == 'criteria':
                        score_response = cur_rephrased_question.replace("Scoring Criteria:", "")
                        proposed_question_dict['scoring-criteria'] = score_response
                    else:
                        raise ValueError("Invalid score_type")

                    new_gen_num += 1
                    if (new_gen_num + 1) % self.save_interval == 0:
                        print(f"Saving results to {os.path.relpath(self.REPHRASE_EVALUATOR_OUTPUT_PATH, PROJECT_ROOT)}")
                        save_json(data, self.REPHRASE_EVALUATOR_OUTPUT_PATH)
                        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

                except Exception as e:
                    print(f"Error processing criteria: {e}")
                    continue

        if new_gen_num or not os.path.exists(self.REPHRASE_EVALUATOR_OUTPUT_PATH):
            print(f"Saving results to {os.path.relpath(self.REPHRASE_EVALUATOR_OUTPUT_PATH, PROJECT_ROOT)}")
            save_json(data, self.REPHRASE_EVALUATOR_OUTPUT_PATH)
            print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

        print(f"Token Usage - Prompt: {self.total_prompt_tokens}, Completion: {self.total_completion_tokens}, Total: {self.total_prompt_tokens + self.total_completion_tokens}")
        return new_gen_num, all_num, self.total_prompt_tokens, self.total_completion_tokens

    def run(self):
        """
        Main method to run the rephrase evaluator.

        Returns:
            Tuple of (new_gen_num, all_num, total_prompt_tokens, total_completion_tokens)
        """
        file_name = os.path.basename(self.REPHRASE_EVALUATOR_INPUT_PATH)
        relative_path = os.path.relpath(self.REPHRASE_EVALUATOR_INPUT_PATH, PROJECT_ROOT)
        print(f"Processing file {relative_path}")

        if "content" in file_name:
            return self.process_file_content()
        elif "entity_graph" in file_name:
            return self.process_file_entity_graph()
        else:
            raise ValueError(f"Unknown file type: {file_name}")


if __name__ == '__main__':
    pass
