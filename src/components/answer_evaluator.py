# Import standard library modules for file operations, JSON handling, and regex
import os
import json
import math
import re
import numpy as np
import argparse
import threading
from tqdm import tqdm
from scipy.stats import entropy
from collections import defaultdict
from scipy.special import rel_entr

from concurrent.futures import ThreadPoolExecutor, as_completed

# Import project-specific utilities
from src.tools.string_utils import read_text_file
from src import PROJECT_ROOT
from src.tools.api import call_api_qwen  
from src.tools.rag_utils import list_to_numbered_string, expand_numbers_and_ranges
from src.tools.json_utils import save_json, load_json



class AnswerEvaluator:
    """
    A class to evaluate answers based on various criteria such as relevance,
    semantic similarity, inferability, and practicality.
    """

    def __init__(self, save_interval=10):
        """
        Initialize the AnswerEvaluator with configuration from environment variables.

        Args:
            save_interval: The interval at which to save intermediate results
        """
        self.PROJECT_ROOT = PROJECT_ROOT

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

        self.RELEVANCE_PROMPT = read_text_file(os.path.join(PROJECT_ROOT, relevance_prompt_path))
        self.SEMANTIC_SIMILARITY_PROMPT = read_text_file(os.path.join(PROJECT_ROOT, semantic_similarity_prompt_path))
        self.INFERABILITY_PROMPT = read_text_file(os.path.join(PROJECT_ROOT, inferability_prompt_path))
        self.PRACTICALITY_PROMPT = read_text_file(os.path.join(PROJECT_ROOT, practicality_prompt_path))

        self.temperature = float(os.getenv("TEMPERATURE", 0.6))
        self.save_interval = save_interval

        # Load paths from environment
        self.ANSWER_EVALUATOR_INPUT_PATH, self.ANSWER_EVALUATOR_OUTPUT_PATH = None, None
        if os.getenv("ANSWER_EVALUATOR_CONTENT_INPUT_PATH", None) != None:
            self.ANSWER_EVALUATOR_INPUT_PATH = os.getenv("ANSWER_EVALUATOR_CONTENT_INPUT_PATH")
            self.ANSWER_EVALUATOR_OUTPUT_PATH = os.getenv("ANSWER_EVALUATOR_CONTENT_OUTPUT_PATH")
        elif os.getenv("ANSWER_EVALUATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
            self.ANSWER_EVALUATOR_INPUT_PATH = os.getenv("ANSWER_EVALUATOR_ENTITYGRAPH_INPUT_PATH")
            self.ANSWER_EVALUATOR_OUTPUT_PATH = os.getenv("ANSWER_EVALUATOR_ENTITYGRAPH_OUTPUT_PATH")
        else:
            raise EnvironmentError("Environment variable 'ANSWER_EVALUATOR_CONTENT_INPUT_PATH' or 'ANSWER_EVALUATOR_ENTITYGRAPH_INPUT_PATH' is not set.")

        self.ANSWER_EVALUATOR_INPUT_PATH = os.path.join(PROJECT_ROOT, self.ANSWER_EVALUATOR_INPUT_PATH)
        self.ANSWER_EVALUATOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.ANSWER_EVALUATOR_OUTPUT_PATH)
        self.ANSWER_EVALUATOR_NUM_WORKERS = int(os.getenv("ANSWER_EVALUATOR_NUM_WORKERS", 4))
        self.ANSWER_EVALUATOR_MAX_GEN_TIMES = int(os.getenv("ANSWER_EVALUATOR_MAX_GEN_TIMES", -1))

        # Token usage tracker
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.token_lock = threading.Lock()

    def extract_reasoning_and_score(self, text):
        try:
            # Use regular expressions to find 'Reasoning' and 'Score' fields in the LLM response
            reasoning_match = re.search(r'Reasoning\s*:\s*(.*)', text, re.IGNORECASE) # Pattern matches "Reasoning:" followed by any text, case-insensitive
            score_match = re.search(r'Score\s*:\s*(.*)', text, re.IGNORECASE) # Pattern matches "Score:" followed by any text (expected to be a number)

            # Check if both 'Reasoning' and 'Score' were found in the response. If neither is found, the response format is invalid
            if not reasoning_match and not score_match:
                return None, None  # Return None to indicate format error

            
            reasoning = reasoning_match.group(1).strip() # Extract the reasoning text (everything after "Reasoning:") and remove whitespace
            score_str = score_match.group(1).strip() # Extract the score text (everything after "Score:") and remove whitespace
            score = int(score_str)
            return reasoning, score
        except ValueError:
            return None, None  

        except Exception:
            return None, None  

    def score_relevance(self, answer, question):
        """
        Scores the RAG's answer based on the Relevance criterion.
        """
        cur_prompt = self.RELEVANCE_PROMPT.format(question=question, answer=answer)
        generator_response, prompt_tokens, completion_tokens, _ = call_api_qwen(cur_prompt, self.temperature)
        return generator_response, prompt_tokens, completion_tokens

    def score_inferability(self, answer, clues, question):
        """
        Scores the RAG's answer based on the Inferability criterion.
        """
        # Format the inferability prompt template with question, answer, and clues
        # Clues are the objective facts that the answer should be inferable from
        cur_prompt = self.INFERABILITY_PROMPT.format(question=question, answer=answer, clues=clues)
        # Call the Qwen API to evaluate if the answer can be reasonably inferred from the clues
        generator_response, prompt_tokens, completion_tokens, _ = call_api_qwen(cur_prompt, self.temperature)
        # Return the evaluation response and token usage statistics
        return generator_response, prompt_tokens, completion_tokens

    def score_practicality(self, answer, question):
        """
        Scores the question based on the Practicality criterion.
        """
        # Format the practicality prompt template with the question and answer
        cur_prompt = self.PRACTICALITY_PROMPT.format(question=question, answer=answer)
        # Call the Qwen API to evaluate the practical value/usefulness of the question
        generator_response, prompt_tokens, completion_tokens, _ = call_api_qwen(cur_prompt, self.temperature)
        # Return the practicality evaluation and token counts
        return generator_response, prompt_tokens, completion_tokens

    def score_semantic_similarity(self, clues, question):
        """
        Scores the semantic similarity between the question and the clues based on the Semantic Similarity criterion.
        """
        # Format the semantic similarity prompt template with question and clues
        cur_prompt = self.SEMANTIC_SIMILARITY_PROMPT.format(question=question, clues=clues)
        # Call the Qwen API to evaluate how semantically similar the question is to the provided clues
        generator_response, prompt_tokens, completion_tokens, _ = call_api_qwen(cur_prompt, self.temperature)
        # Return the similarity evaluation and token usage information
        return generator_response, prompt_tokens, completion_tokens

    def process_file_content(self):
        """
        Process content-based files for answer evaluation.
        """
        os.makedirs(os.path.dirname(self.ANSWER_EVALUATOR_OUTPUT_PATH), exist_ok=True)

        # If output file already exists, resume from saved progress. Otherwise, start fresh from the input file
        if os.path.exists(self.ANSWER_EVALUATOR_OUTPUT_PATH):
            data = load_json(self.ANSWER_EVALUATOR_OUTPUT_PATH)
        else:
            data = load_json(self.ANSWER_EVALUATOR_INPUT_PATH)

        answer_evaluator_max_gen_times = self.ANSWER_EVALUATOR_MAX_GEN_TIMES
        if answer_evaluator_max_gen_times == -1: # If max_gen_times is -1, process all items in the dataset
            answer_evaluator_max_gen_times = len(data) # Otherwise, limit to the specified number


        all_num, new_gen_num = 0, 0 # Initialize counters: all_num = total tasks, new_gen_num = newly processed tasks

        with ThreadPoolExecutor(max_workers=self.ANSWER_EVALUATOR_NUM_WORKERS) as executor:
            futures_to_data = {} # Dictionary to map Future objects to their associated data
            for cur_dict in data[:answer_evaluator_max_gen_times]: 

                """calculate generation metrics"""
                chunk_id = cur_dict['id']
                objective_facts = cur_dict['objective-facts'] # Get the list of objective facts extracted from this content chunk
                objective_fact_id_2_objective_prompt = {idx: fact for idx, fact in enumerate(objective_facts, start=1)} # Create a mapping from fact ID (1-indexed) to the actual fact text

                
                if 'proposed-questions' not in cur_dict: # Skip this chunk if it doesn't have any proposed questions
                    continue
                proposed_questions = cur_dict['proposed-questions'] # Get the dictionary of proposed questions for this chunk
                if_already_generated = False # Flag to track if all scoring is already complete (currently unused - always False)
               
                for question_type, question_dict in proposed_questions.items():
                    question = question_dict['question'] # Extract the question text
                    if "objective-facts" in question_dict: # Parse the objective-facts field 
                        objective_fact_clue_ids = re.findall(r'\d+-\d+|\d+', question_dict['objective-facts'].strip()) # Find all numbers and ranges (e.g., "1-3, 5" -> ["1-3", "5"])
                        objective_fact_clue_ids = expand_numbers_and_ranges(objective_fact_clue_ids) # Expand ranges into individual numbers (e.g., "1-3" -> ["1", "2", "3"])
                    else:
                        objective_fact_clue_ids = []

                    clues = [objective_fact_id_2_objective_prompt[int(clue_id)] for clue_id in objective_fact_clue_ids if int(clue_id) in objective_fact_id_2_objective_prompt]
                    clues_str = list_to_numbered_string(clues)

                    # Find all answer keys in this question (e.g., "answer", "rag-answer")
                    # Exclude keys that contain 'score' or 'reason' (those are evaluation results)
                    answer_keys = [key for key in question_dict.keys() if 'answer' in key and 'score' not in key and 'reason' not in key]

                    # For each answer, check if it needs to be scored
                    for answer_key in answer_keys:
                        
                        answer = question_dict[answer_key] # Get the answer text

                        # Construct the key name where the relevance score will be stored
                        relevance_score_key_name = f"{answer_key}-relevance-score"

                        # Relevance Score: only evaluate if not already scored
                        if relevance_score_key_name not in question_dict or question_dict[relevance_score_key_name] == None:
                            # Submit the scoring task to the thread pool for parallel execution
                            future = executor.submit(self.score_relevance, answer, question)
                            # Store metadata to identify this task when it completes
                            futures_to_data[future] = (question_dict, answer_key, 'relevance')

                if if_already_generated: 
                    continue

            # Count total number of scoring tasks submitted
            all_num = len(futures_to_data)
            for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
                question_dict, answer_key, score_type = futures_to_data[future] # Retrieve the metadata associated with this future
                try:
                    score_response, prompt_tokens, completion_tokens = future.result(timeout=10*60) # Get the result from the completed future with a 10-minute timeout

                    with self.token_lock:
                        self.total_prompt_tokens += prompt_tokens
                        self.total_completion_tokens += completion_tokens
 
                    reason, score = self.extract_reasoning_and_score(score_response) # Parse the LLM response to extract reasoning and numeric score
                    # Construct the keys where results will be stored in the data structure
                    score_key_name = f"{answer_key}-{score_type}-score"
                    reason_key_name = f"{answer_key}-{score_type}-reason"

                    # Store the reasoning and score in the question dictionary
                    question_dict[reason_key_name] = reason
                    question_dict[score_key_name] = score

                    # Increment the counter of newly processed tasks
                    new_gen_num += 1
                    # Periodically save progress to avoid losing work if the process crashes
                    if (new_gen_num + 1) % self.save_interval == 0:
                        print(f"Saving results to {os.path.relpath(self.ANSWER_EVALUATOR_OUTPUT_PATH, PROJECT_ROOT)}")
                        save_json(data, self.ANSWER_EVALUATOR_OUTPUT_PATH)
                        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

                except Exception as e:
                    # Log the error and continue processing other futures
                    print(f"Error processing {score_type} for answer_key {answer_key}: {e}")
                    continue

        # Final save: save if any new results were generated OR if output file doesn't exist yet
        if new_gen_num or not os.path.exists(self.ANSWER_EVALUATOR_OUTPUT_PATH):
            print(f"Saving results to {os.path.relpath(self.ANSWER_EVALUATOR_OUTPUT_PATH, PROJECT_ROOT)}")
            save_json(data, self.ANSWER_EVALUATOR_OUTPUT_PATH)
            print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

        # Print token usage statistics
        print(f"Token Usage - Prompt: {self.total_prompt_tokens}, Completion: {self.total_completion_tokens}, Total: {self.total_prompt_tokens + self.total_completion_tokens}")

        # Return statistics: number of newly processed tasks, total tasks, and token counts
        return new_gen_num, all_num, self.total_prompt_tokens, self.total_completion_tokens

    def process_file_entity_graph(self):
        """
        Process entity graph-based files for answer evaluation.
        """
        os.makedirs(os.path.dirname(self.ANSWER_EVALUATOR_OUTPUT_PATH), exist_ok=True)

        # If output file already exists, resume from saved progress
        # Otherwise, start fresh from the input file
        if os.path.exists(self.ANSWER_EVALUATOR_OUTPUT_PATH):
            data = load_json(self.ANSWER_EVALUATOR_OUTPUT_PATH)
        else:
            data = load_json(self.ANSWER_EVALUATOR_INPUT_PATH)

        # If max_gen_times is -1, process all entities in the dataset
        # Otherwise, limit to the specified number
        answer_evaluator_max_gen_times = self.ANSWER_EVALUATOR_MAX_GEN_TIMES
        if answer_evaluator_max_gen_times == -1:
            answer_evaluator_max_gen_times = len(data)

        all_num, new_gen_num = 0, 0
        with ThreadPoolExecutor(max_workers=self.ANSWER_EVALUATOR_NUM_WORKERS) as executor:
            
            futures_to_data = {} # Dictionary to map Future objects to their associated data
            # Iterate through entities up to the specified limit
            # Note: data is a dict, so we convert items() to a list before slicing
            for entity_id, entity_dict in list(data.items())[:answer_evaluator_max_gen_times]:

                proposed_questions = entity_dict['proposed-questions'] # Get the dictionary of proposed questions for this entity

                # Get the objective relationships (entity connections) that are relevant to this entity
                objective_relationships = entity_dict['selected-relationships']['objective-relationships']
                # Create a mapping from relationship ID (1-indexed) to the actual relationship text
                objective_fact_id_2_objective_relationship = {idx: fact for idx, fact in enumerate(objective_relationships, start=1)}

                """calculate generation metrics"""
                # Flag to track if all scoring is already complete (currently unused - always False)
                if_already_generated = False
                # Iterate through each question type for this entity
                for question_type, question_dict in proposed_questions.items():
                    # Extract the question text
                    question = question_dict['question']

                    # Parse the objective-relationship-id field to get which relationships are relevant
                    # Find all numbers and ranges (e.g., "1-3, 5" -> ["1-3", "5"])
                    objective_relationship_ids = re.findall(r'\d+-\d+|\d+', question_dict['objective-relationship-id'].strip())
                    # Expand ranges into individual numbers (e.g., "1-3" -> ["1", "2", "3"])
                    objective_relationship_ids = expand_numbers_and_ranges(objective_relationship_ids)

                    # Get the answer for this question
                    answer = question_dict['answer']

                    # First, identify which relationships correspond to the correct answers, and then locate the relevant documents based on those relationships.
                    # Retrieve the actual relationship texts corresponding to the IDs
                    real_related_relationships = [objective_fact_id_2_objective_relationship[int(clue_id)] for clue_id in objective_relationship_ids if int(clue_id) in objective_fact_id_2_objective_relationship]

                    # Find all answer keys in this question (e.g., "answer", "rag-answer")
                    # Exclude keys that contain 'score' or 'reason' (those are evaluation results)
                    answer_keys = [key for key in question_dict.keys() if 'answer' in key and 'score' not in key and 'reason' not in key]
                    # For each answer, check if it needs to be scored
                    for answer_key in answer_keys:
                        # Get the answer text
                        answer = question_dict[answer_key]

                        # Construct the key name where the relevance score will be stored
                        relevance_score_key_name = f"{answer_key}-relevance-score"

                        # Relevance Score: only evaluate if not already scored
                        if relevance_score_key_name not in question_dict or question_dict[relevance_score_key_name] == None:
                            # Submit the scoring task to the thread pool for parallel execution
                            future = executor.submit(self.score_relevance, answer, question)
                            # Store metadata to identify this task when it completes
                            futures_to_data[future] = (question_dict, answer_key, 'relevance')

                # Skip to next entity if already generated (this condition is never true currently)
                if if_already_generated:
                    continue

            all_num = len(futures_to_data)
            for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
                question_dict, answer_key, score_type = futures_to_data[future] # Retrieve the metadata associated with this future
                try:
                    score_response, prompt_tokens, completion_tokens = future.result(timeout=10*60) # Get the result from the completed future with a 10-minute timeout

                    with self.token_lock:
                        self.total_prompt_tokens += prompt_tokens
                        self.total_completion_tokens += completion_tokens

                    
                    reason, score = self.extract_reasoning_and_score(score_response) # Parse the LLM response to extract reasoning and numeric score
                    # Construct the keys where results will be stored in the data structure
                    score_key_name = f"{answer_key}-{score_type}-score"
                    reason_key_name = f"{answer_key}-{score_type}-reason"

                    # Store the reasoning and score in the question dictionary
                    question_dict[reason_key_name] = reason
                    question_dict[score_key_name] = score

                    # Increment the counter of newly processed tasks
                    new_gen_num += 1
                    # Periodically save progress to avoid losing work if the process crashes
                    if (new_gen_num + 1) % self.save_interval == 0:
                        print(f"Saving results to {os.path.relpath(self.ANSWER_EVALUATOR_OUTPUT_PATH, PROJECT_ROOT)}")
                        save_json(data, self.ANSWER_EVALUATOR_OUTPUT_PATH)
                        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

                except Exception as e:
                    # Log the error and continue processing other futures
                    print(f"Error processing {score_type} for answer_key {answer_key}: {e}")
                    continue

        # Final save: save if any new results were generated OR if output file doesn't exist yet
        if new_gen_num or not os.path.exists(self.ANSWER_EVALUATOR_OUTPUT_PATH):
            print(f"Saving results to {os.path.relpath(self.ANSWER_EVALUATOR_OUTPUT_PATH, PROJECT_ROOT)}")
            save_json(data, self.ANSWER_EVALUATOR_OUTPUT_PATH)
            print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

        # Print token usage statistics
        print(f"Token Usage - Prompt: {self.total_prompt_tokens}, Completion: {self.total_completion_tokens}, Total: {self.total_prompt_tokens + self.total_completion_tokens}")

        # Return statistics: number of newly processed tasks, total tasks, and token counts
        return new_gen_num, all_num, self.total_prompt_tokens, self.total_completion_tokens

    def run(self):
        """
        Main method to run the answer evaluator.

        Returns:
            Tuple of (new_gen_num, all_num, total_prompt_tokens, total_completion_tokens)
        """
        # Extract just the filename from the full path
        file_name = os.path.basename(self.ANSWER_EVALUATOR_INPUT_PATH)
        # Get the path relative to project root for cleaner logging
        relative_path = os.path.relpath(self.ANSWER_EVALUATOR_INPUT_PATH, PROJECT_ROOT)
        # Log which file is being processed
        print(f"Processing file {relative_path}")

        # Determine file type based on filename and route to appropriate processing method
        # Content-based files contain text chunks with objective facts
        if "content" in file_name:
            return self.process_file_content()
        # Entity graph files contain entity relationships and graph-based questions
        elif "entity_graph" in file_name:
            return self.process_file_entity_graph()
        else:
            # Raise an error if the file type cannot be determined from the filename
            raise ValueError(f"Unknown file type: {file_name}")


if __name__ == '__main__':
    pass