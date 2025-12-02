import os
import json
import math
import re
import numpy as np
import argparse
from tqdm import tqdm
from openai import OpenAI
from scipy.stats import entropy
from collections import defaultdict
from scipy.special import rel_entr
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.tools.string_utils import read_text_file
from src import PROJECT_ROOT
from rag.utils.request_openai_utils import OpenAIModel
from src.tools.rag_utils import list_to_numbered_string, expand_numbers_and_ranges


class AnswerEvaluator:
    """
    A class to evaluate answers based on various criteria such as relevance,
    semantic similarity, inferability, and practicality.
    """

    def __init__(self, save_interval):
        """
        Initialize the AnswerEvaluator with configuration from environment variables.

        Args:
            save_interval: The interval at which to save intermediate results
        """
        # Use PROJECT_ROOT for paths
        self.PROJECT_ROOT = PROJECT_ROOT

        # Load prompt paths from environment variables
        relevance_prompt_path = os.getenv("ANSWER_EVALUATOR_RELEVANCE_PROMPT_PATH", None)
        semantic_similarity_prompt_path = os.getenv("ANSWER_EVALUATOR_SEMANTIC_SIMILARITY_PROMPT_PATH", None)
        inferability_prompt_path = os.getenv("ANSWER_EVALUATOR_INFERABILITY_PROMPT_PATH", None)
        practicality_prompt_path = os.getenv("ANSWER_EVALUATOR_PRACTICALITY_PROMPT_PATH", None)

        if relevance_prompt_path is None:
            raise ValueError("ANSWER_EVALUATOR_RELEVANCE_PROMPT_PATH environment variable is not set")
        if semantic_similarity_prompt_path is None:
            raise ValueError("ANSWER_EVALUATOR_SEMANTIC_SIMILARITY_PROMPT_PATH environment variable is not set")
        if inferability_prompt_path is None:
            raise ValueError("ANSWER_EVALUATOR_INFERABILITY_PROMPT_PATH environment variable is not set")
        if practicality_prompt_path is None:
            raise ValueError("ANSWER_EVALUATOR_PRACTICALITY_PROMPT_PATH environment variable is not set")

        # Load prompts from files
        self.RELEVANCE_PROMPT = read_text_file(os.path.join(PROJECT_ROOT, relevance_prompt_path))
        self.SEMANTIC_SIMILARITY_PROMPT = read_text_file(os.path.join(PROJECT_ROOT, semantic_similarity_prompt_path))
        self.INFERABILITY_PROMPT = read_text_file(os.path.join(PROJECT_ROOT, inferability_prompt_path))
        self.PRACTICALITY_PROMPT = read_text_file(os.path.join(PROJECT_ROOT, practicality_prompt_path))

        # Load OpenAI configuration
        api_key = os.getenv("API_KEY", "None")
        base_url = os.getenv("BASE_URL", None)

        if base_url is None:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI(base_url=f"{base_url}", api_key=api_key)

        model_name = os.getenv("MODEL_NAME", None)
        if model_name is None:
            raise EnvironmentError("MODEL_NAME environment variable is not set")

        stop_words = "------"
        max_new_tokens = "None"
        self.openai_model = OpenAIModel(model_name, stop_words, max_new_tokens)
        self.temperature = float(os.getenv("TEMPERATURE", 0.6))

        self.save_interval = save_interval

    def extract_reasoning_and_score(self, text):
        """
        Extracts 'Reasoning' and 'Score' from the input text.
        Converts 'Score' to int and keeps 'Reasoning' as str.
        If extraction fails or format is incorrect, returns None.
        """
        try:
            # Use regular expressions to find 'Reasoning' and 'Score'
            reasoning_match = re.search(r'Reasoning\s*:\s*(.*)', text, re.IGNORECASE)
            score_match = re.search(r'Score\s*:\s*(.*)', text, re.IGNORECASE)

            # Check if both 'Reasoning' and 'Score' were found
            if not reasoning_match and not score_match:
                return None, None  # Return None to indicate format error

            # Extract the reasoning and score string
            reasoning = reasoning_match.group(1).strip()
            score_str = score_match.group(1).strip()

            # Convert 'Score' to int
            score = int(score_str)

            return reasoning, score

        except ValueError:
            # Score is not a valid integer
            return None, None  # Return None to indicate format error

        except Exception:
            # Catch any other exceptions
            return None, None  # Return None to indicate format error

    def score_relevance(self, answer, question):
        """
        Scores the RAG's answer based on the Relevance criterion.
        """
        cur_prompt = self.RELEVANCE_PROMPT.format(question=question, answer=answer)
        generator_response, _ = self.openai_model.generate(self.client, cur_prompt, self.temperature)
        return generator_response

    # def score_inferability(self, answer, clues, question):
    #     """
    #     Scores the RAG's answer based on the Inferability criterion.
    #     """
    #     cur_prompt = self.INFERABILITY_PROMPT.format(question=question, answer=answer, clues=clues)
    #     generator_response, _ = self.openai_model.generate(self.client, cur_prompt, self.temperature)
    #     return generator_response

    # def score_practicality(self, answer, question):
    #     """
    #     Scores the question based on the Practicality criterion.
    #     """
    #     cur_prompt = self.PRACTICALITY_PROMPT.format(question=question, answer=answer)
    #     generator_response, _ = self.openai_model.generate(self.client, cur_prompt, self.temperature)
    #     return generator_response

    def score_semantic_similarity(self, clues, question):
        """
        Scores the semantic similarity between the question and the clues based on the Semantic Similarity criterion.
        """
        cur_prompt = self.SEMANTIC_SIMILARITY_PROMPT.format(question=question, clues=clues)
        generator_response, _ = self.openai_model.generate(self.client, cur_prompt, self.temperature)
        return generator_response

    def process_file_content(self, input_path, output_path, max_workers, answer_evaluator_max_gen_times):
        """
        Process content-based files for answer evaluation.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                data = json.load(f)
        else:
            with open(input_path, 'r') as f:
                data = json.load(f)

        if answer_evaluator_max_gen_times == -1:
            answer_evaluator_max_gen_times = len(data)

        all_num, new_gen_num = 0, 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_to_data = {}
            for cur_dict in data[:answer_evaluator_max_gen_times]:

                """calculate generation metrics"""
                chunk_id = cur_dict['id']  # admission.stanford.edu.filter_index.htm.md_chunk_0
                objective_facts = cur_dict['objective-facts']
                objective_fact_id_2_objective_prompt = {idx: fact for idx, fact in enumerate(objective_facts, start=1)}

                if 'proposed-questions' not in cur_dict:
                    continue
                proposed_questions = cur_dict['proposed-questions']
                if_already_generated = False
                for question_type, question_dict in proposed_questions.items():

                    question = question_dict['question']

                    if "objective-facts" in question_dict:
                        objective_fact_clue_ids = re.findall(r'\d+-\d+|\d+', question_dict['objective-facts'].strip())
                        objective_fact_clue_ids = expand_numbers_and_ranges(objective_fact_clue_ids)
                    else:
                        objective_fact_clue_ids = []

                    clues = [objective_fact_id_2_objective_prompt[int(clue_id)] for clue_id in objective_fact_clue_ids if int(clue_id) in objective_fact_id_2_objective_prompt]
                    clues_str = list_to_numbered_string(clues)

                    answer_keys = [key for key in question_dict.keys() if 'answer' in key and 'score' not in key and 'reason' not in key]

                    for answer_key in answer_keys:
                        answer = question_dict[answer_key]

                        relevance_score_key_name = f"{answer_key}-relevance-score"

                        # Relevance Score
                        if relevance_score_key_name not in question_dict or question_dict[relevance_score_key_name] == None:
                            future = executor.submit(self.score_relevance, answer, question)
                            futures_to_data[future] = (question_dict, answer_key, 'relevance')

                if if_already_generated:
                    continue

            all_num = len(futures_to_data)
            for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
                question_dict, answer_key, score_type = futures_to_data[future]
                try:
                    score_response = future.result(timeout=10*60)
                    reason, score = self.extract_reasoning_and_score(score_response)
                    score_key_name = f"{answer_key}-{score_type}-score"
                    reason_key_name = f"{answer_key}-{score_type}-reason"

                    question_dict[reason_key_name] = reason
                    question_dict[score_key_name] = score

                    new_gen_num += 1
                    if (new_gen_num + 1) % self.save_interval == 0:
                        print(f"Saving results to {output_path}")
                        with open(output_path, 'w') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

                except Exception as e:
                    print(f"Error processing {score_type} for answer_key {answer_key}: {e}")
                    continue

        if new_gen_num or not os.path.exists(output_path):
            print(f"Saving results to {output_path}")
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

        return new_gen_num, all_num

    def process_file_entity_graph(self, input_path, output_path, max_workers, answer_evaluator_max_gen_times):
        """
        Process entity graph-based files for answer evaluation.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                data = json.load(f)
        else:
            with open(input_path, 'r') as f:
                data = json.load(f)

        if answer_evaluator_max_gen_times == -1:
            answer_evaluator_max_gen_times = len(data)

        all_num, new_gen_num = 0, 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_to_data = {}

            for entity_id, entity_dict in list(data.items())[:answer_evaluator_max_gen_times]:

                proposed_questions = entity_dict['proposed-questions']

                objective_relationships = entity_dict['selected-relationships']['objective-relationships']
                objective_fact_id_2_objective_relationship = {idx: fact for idx, fact in enumerate(objective_relationships, start=1)}

                """calculate generation metrics"""
                if_already_generated = False
                for question_type, question_dict in proposed_questions.items():
                    question = question_dict['question']

                    objective_relationship_ids = re.findall(r'\d+-\d+|\d+', question_dict['objective-relationship-id'].strip())
                    objective_relationship_ids = expand_numbers_and_ranges(objective_relationship_ids)

                    answer = question_dict['answer']

                    # First, identify which relationships correspond to the correct answers, and then locate the relevant documents based on those relationships.
                    real_related_relationships = [objective_fact_id_2_objective_relationship[int(clue_id)] for clue_id in objective_relationship_ids if int(clue_id) in objective_fact_id_2_objective_relationship]

                    answer_keys = [key for key in question_dict.keys() if 'answer' in key and 'score' not in key and 'reason' not in key]
                    for answer_key in answer_keys:
                        answer = question_dict[answer_key]

                        relevance_score_key_name = f"{answer_key}-relevance-score"

                        # Relevance Score
                        if relevance_score_key_name not in question_dict or question_dict[relevance_score_key_name] == None:
                            future = executor.submit(self.score_relevance, answer, question)
                            futures_to_data[future] = (question_dict, answer_key, 'relevance')

                if if_already_generated:
                    continue

            all_num = len(futures_to_data)
            for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
                question_dict, answer_key, score_type = futures_to_data[future]
                try:
                    score_response = future.result(timeout=10*60)
                    reason, score = self.extract_reasoning_and_score(score_response)
                    score_key_name = f"{answer_key}-{score_type}-score"
                    reason_key_name = f"{answer_key}-{score_type}-reason"

                    question_dict[reason_key_name] = reason
                    question_dict[score_key_name] = score

                    new_gen_num += 1
                    if (new_gen_num + 1) % self.save_interval == 0:
                        print(f"Saving results to {output_path}")
                        with open(output_path, 'w') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

                except Exception as e:
                    print(f"Error processing {score_type} for answer_key {answer_key}: {e}")
                    continue

        if new_gen_num or not os.path.exists(output_path):
            print(f"Saving results to {output_path}")
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

        return new_gen_num, all_num

    def run(self, input_path, output_path, max_workers, answer_evaluator_max_gen_times):
        """
        Main method to run the answer evaluator.

        Args:
            input_path: Path to input file
            output_path: Path to output file
            max_workers: Maximum number of concurrent workers
            answer_evaluator_max_gen_times: Maximum number of items to process (-1 for all)

        Returns:
            Tuple of (new_gen_num, all_num)
        """
        file_name = os.path.basename(input_path)
        relative_path = os.path.relpath(input_path, self.PROJECT_ROOT)
        print(f"Processing file {relative_path}")

        if "content" in file_name:
            return self.process_file_content(input_path, output_path, max_workers, answer_evaluator_max_gen_times)
        elif "entity_graph" in file_name:
            return self.process_file_entity_graph(input_path, output_path, max_workers, answer_evaluator_max_gen_times)
        else:
            raise ValueError(f"Unknown file type: {file_name}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Calculate metrics for answer.")
    parser.add_argument('--input_path', type=str, help="Input file containing query results.")
    parser.add_argument('--output_path', type=str, help="Output file to save the results.")
    parser.add_argument('--save_interval', type=int, help="The interval at which to save the results.")
    parser.add_argument('--max_workers', type=int, default=8, help="Maximum number of concurrent requests.")
    parser.add_argument('--answer_evaluator_max_gen_times', type=int, default=-1, help="Maximum number of items to process (-1 for all).")
    args = parser.parse_args()

    # Use PROJECT_ROOT for paths
    args.input_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.input_path))
    args.output_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.output_path))

    # Create an instance and run
    evaluator = AnswerEvaluator(save_interval=args.save_interval)
    evaluator.run(args.input_path, args.output_path, args.max_workers, args.answer_evaluator_max_gen_times)
