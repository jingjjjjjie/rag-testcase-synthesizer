#You can't manually write scoring criteria for thousands of automatically generated questions. So this component automatically generates the evaluation rubric by asking an LLM: "Given this original question, this transformation, and this answer - what should we check to see if a RAG system answered it correctly?" In short: It creates the "answer key" for grading RAG systems on transformed questions.

import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import re

from src.tools.api import call_api_qwen
from src.tools.rag_utils import expand_numbers_and_ranges
from ..import PROJECT_ROOT
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

# Path to prompts directory
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "src", "prompts")

def load_prompt(prompt_type: str) -> str:
    """Load prompt from file."""
    prompt_file = os.path.join(PROMPTS_DIR, f"{prompt_type}_evaluator.txt")
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def scoring_by_points_once_give_one(transformed_question, rephrased_type: Optional[str] = ['rephrased', 'rephrased_part', 'rephrased_hybrid', 'rephrased_hybrid_part']):

    if rephrased_type == 'rephrased':
        cur_prompt = load_prompt('rephrased')

    elif rephrased_type == 'rephrased_part':
        cur_prompt = load_prompt('rephrased_part')
    elif rephrased_type == 'rephrased_hybrid':
        cur_prompt = load_prompt('rephrased_hybrid')
    elif rephrased_type == 'rephrased_hybrid_part':
        cur_prompt = load_prompt('rephrased_hybrid_part')
    else:
        raise ValueError("Invalid rephrased_type")

    cur_prompt = cur_prompt.replace("[[TRANSFORMED QUESTION]]", transformed_question)

    response, prompt_tokens, completion_tokens, _ = call_api_qwen(cur_prompt, TEMPERATURE)
    return response, prompt_tokens, completion_tokens

def get_transformed_question(last_question, last_answer, cur_rephrased_question):
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

def process_file_content(input_path, output_path, save_interval, max_workers, rephrase_evaluator_max_gen_times):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
    else:
        with open(input_path, 'r') as f:
            data = json.load(f)

    if rephrase_evaluator_max_gen_times == -1:
        rephrase_evaluator_max_gen_times = len(data)

    all_num, new_gen_num = 0, 0
    total_prompt_tokens, total_completion_tokens = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_data = {}
        for cur_dict in data[:rephrase_evaluator_max_gen_times]:
            
            if 'proposed-questions' not in cur_dict:
                continue
            proposed_questions = cur_dict['proposed-questions']

            if_already_generated = False
            for proposed_question_type, proposed_question_dict in proposed_questions.items():

                question = proposed_question_dict['question']
                answer = proposed_question_dict['answer']
                
                # cur_rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                # last_question = question
                # last_answer = answer
                # for cur_rephrased_question in cur_rephrased_questions:
                #     if 'scoring-criteria' in cur_rephrased_question:
                #         continue
                #     transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question)
                #     future = executor.submit(scoring_by_points_once_give_one, CLIENT, transformed_question, rephrased_type='rephrased')
                #     futures_to_data[future] = (cur_rephrased_question, 'criteria')
                #     last_question = cur_rephrased_question['result']
                #     last_answer = cur_rephrased_question['answer']
                
                cur_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                last_question = question
                last_answer = answer
                for cur_rephrased_question_part in cur_rephrased_questions_part:
                    if 'scoring-criteria' in cur_rephrased_question_part:
                        continue
                    transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question_part)
                    # print("transformed_question:", transformed_question)
                    # input("Press Enter to continue...")
                    future = executor.submit(scoring_by_points_once_give_one, transformed_question, rephrased_type='rephrased_part')
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
                        transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question_hybrid)
                        future = executor.submit(scoring_by_points_once_give_one, transformed_question, rephrased_type='rephrased_hybrid_part')
                        futures_to_data[future] = (cur_rephrased_question_hybrid, 'criteria')
                    last_question = cur_rephrased_question_hybrid['result']
                    last_answer = cur_rephrased_question_hybrid['answer']

            if if_already_generated:
                continue

        all_num = len(futures_to_data)
        for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
            cur_rephrased_question, score_type = futures_to_data[future]
            score_response, prompt_tokens, completion_tokens = future.result(timeout=5*60)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            if score_type == 'criteria':
                score_response = score_response.replace("Scoring Criteria:", "")
                cur_rephrased_question['scoring-criteria'] = score_response
            else:
                raise ValueError("Invalid score_type")

            new_gen_num += 1
            if (new_gen_num + 1) % save_interval == 0:
                print(f"Saving results to {output_path}")
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

    if new_gen_num or not os.path.exists(output_path):
        print(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

    print(f"Token Usage - Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens}, Total: {total_prompt_tokens + total_completion_tokens}")
    return new_gen_num, all_num, total_prompt_tokens, total_completion_tokens

def process_file_entity_graph(input_path, output_path, save_interval, max_workers, rephrase_evaluator_max_gen_times):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
    else:
        with open(input_path, 'r') as f:
            data = json.load(f)

    if rephrase_evaluator_max_gen_times == -1:
        rephrase_evaluator_max_gen_times = len(data)

    all_num, new_gen_num = 0, 0
    total_prompt_tokens, total_completion_tokens = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_data = {}

        for entity_id, entity_dict in list(data.items())[:rephrase_evaluator_max_gen_times]:
            
            proposed_questions = entity_dict['proposed-questions']
            
            if_already_generated = False
            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                question = proposed_question_dict['question']
                answer = proposed_question_dict['positive']

                # cur_rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                # last_question = question
                # last_answer = answer
                # for cur_rephrased_question in cur_rephrased_questions:
                #     if 'scoring-criteria' in cur_rephrased_question:
                #         continue
                #     transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question)
                #     future = executor.submit(scoring_by_points_once_give_one, CLIENT, transformed_question, rephrased_type='rephrased')
                #     futures_to_data[future] = (cur_rephrased_question, 'criteria')
                #     last_question = cur_rephrased_question['result']
                #     last_answer = cur_rephrased_question['answer']
                
                cur_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                last_question = question
                last_answer = answer
                for cur_rephrased_question_part in cur_rephrased_questions_part:
                    if 'scoring-criteria' in cur_rephrased_question_part:
                        continue
                    transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question_part)
                    future = executor.submit(scoring_by_points_once_give_one, transformed_question, rephrased_type='rephrased_part')
                    futures_to_data[future] = (cur_rephrased_question_part, 'criteria')
                    last_question = cur_rephrased_question_part['result']
                    last_answer = cur_rephrased_question_part['answer']
                
                cur_rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                last_question = question
                last_answer = answer
                for cur_rephrased_question_hybrid in cur_rephrased_questions_hybrid:
                    if 'scoring-criteria' in cur_rephrased_question_hybrid:
                        continue
                    transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question_hybrid)
                    future = executor.submit(scoring_by_points_once_give_one, transformed_question, rephrased_type='rephrased_hybrid_part')
                    futures_to_data[future] = (cur_rephrased_question_hybrid, 'criteria')
                    last_question = cur_rephrased_question_hybrid['result']
                    last_answer = cur_rephrased_question_hybrid['answer']

            if if_already_generated:
                continue

        all_num = len(futures_to_data)
        for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
            proposed_question_dict, score_type = futures_to_data[future]
            cur_rephrased_question, prompt_tokens, completion_tokens = future.result(timeout=10*60)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            if score_type == 'criteria':
                score_response = cur_rephrased_question.replace("Scoring Criteria:", "")
                proposed_question_dict['scoring-criteria'] = score_response
            else:
                raise ValueError("Invalid score_type")

            new_gen_num += 1
            if (new_gen_num + 1) % save_interval == 0:
                print(f"Saving results to {output_path}")
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

    if new_gen_num or not os.path.exists(output_path):
        print(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

    print(f"Token Usage - Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens}, Total: {total_prompt_tokens + total_completion_tokens}")
    return new_gen_num, all_num, total_prompt_tokens, total_completion_tokens
    
def rephrase_evaluator(input_path, output_path, save_interval, max_workers, rephrase_evaluator_max_gen_times):
    file_name = os.path.basename(input_path)
    relative_path = os.path.relpath(input_path, PROJECT_ROOT)
    print(f"Processing file {relative_path}")

    if "content" in file_name:
        return process_file_content(input_path, output_path, save_interval, max_workers, rephrase_evaluator_max_gen_times)
    elif "entity_graph" in file_name:
        return process_file_entity_graph(input_path, output_path, save_interval, max_workers, rephrase_evaluator_max_gen_times)
    else:
        raise ValueError(f"Unknown file type: {file_name}")
    
if __name__ == '__main__':
    pass