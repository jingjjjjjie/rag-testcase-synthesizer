import re
import os
import json
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src import PROJECT_ROOT
from src.tools.json_utils import load_json,save_json
from src.tools.string_utils import read_text_file
from src.tools.api import call_api_qwen
from src.tools.rag_utils import (
    reformat_objective_facts,
    extract_largest_json,
    convert_set_to_list, 
    expand_numbers_and_ranges, 
    replace_clue_with_doc_and_sen,
    list_to_docided_string)


TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

def extract_answers(prompt):
    # Define regular expressions to capture the short and long answers
    short_answer_pattern = r"<answer-short>\s*<reason>(.*?)</reason>\s*<answer>(.*?)</answer>\s*</answer-short>"
    long_answer_pattern = r"<answer-long>\s*<reason>(.*?)</reason>\s*<answer>(.*?)</answer>\s*</answer-long>"

    # Search for the patterns in the prompt
    short_answer_match = re.search(short_answer_pattern, prompt, re.DOTALL)
    long_answer_match = re.search(long_answer_pattern, prompt, re.DOTALL)

    # Extract the reason and answer for short and long answers
    if short_answer_match:
        short_reason = short_answer_match.group(1).strip()
        short_answer = short_answer_match.group(2).strip()
    else:
        short_reason = None
        short_answer = None

    if long_answer_match:
        long_reason = long_answer_match.group(1).strip()
        long_answer = long_answer_match.group(2).strip()
    else:
        long_reason = None
        long_answer = None

    return {
        "short-answer": {
            "reason": short_reason,
            "answer": short_answer
        },
        "long-answer": {
            "reason": long_reason,
            "answer": long_answer
        }
    }

class FinalAnswerGenerator:
    def __init__(self, save_interval=10):
        self.FINAL_ANSWER_GENERATOR_INPUT_PATH, self.FINAL_ANSWER_GENERATOR_CORPUS_PATH, self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH = None, None, None
        self.FINAL_ANSWER_GENERATOR_PROMPT_PATH = None
        if os.getenv("FINAL_ANSWER_GENERATOR_CONTENT_INPUT_PATH", None) != None:
            self.FINAL_ANSWER_GENERATOR_INPUT_PATH = os.getenv("FINAL_ANSWER_GENERATOR_CONTENT_INPUT_PATH")
            self.FINAL_ANSWER_GENERATOR_CORPUS_PATH = os.getenv("FINAL_ANSWER_GENERATOR_CORPUS_PATH")
            self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH = os.getenv("FINAL_ANSWER_GENERATOR_CONTENT_OUTPUT_PATH")
            self.FINAL_ANSWER_GENERATOR_PROMPT_PATH = os.getenv("FINAL_ANSWER_GENERATOR_PROMPT_PATH")
            self.FINAL_ANSWER_GENERATOR_GENERATED_TYPE = 'content'
        elif os.getenv("FINAL_ANSWER_GENERATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
            self.FINAL_ANSWER_GENERATOR_INPUT_PATH = os.getenv("FINAL_ANSWER_GENERATOR_ENTITYGRAPH_INPUT_PATH")
            self.FINAL_ANSWER_GENERATOR_CORPUS_PATH = os.getenv("FINAL_ANSWER_GENERATOR_CORPUS_PATH")
            self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH = os.getenv("FINAL_ANSWER_GENERATOR_ENTITYGRAPH_OUTPUT_PATH")
            self.FINAL_ANSWER_GENERATOR_PROMPT_PATH = os.getenv("FINAL_ANSWER_GENERATOR_PROMPT_PATH")
            self.FINAL_ANSWER_GENERATOR_GENERATED_TYPE = 'entity_graph'
        else:
            raise EnvironmentError("Environment variable 'FINAL_ANSWER_GENERATOR_CONTENT_INPUT_PATH' or 'FINAL_ANSWER_GENERATOR_ENTITYGRAPH_INPUT_PATH' is not set.")
        # token usage tracker
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.token_lock = threading.Lock()

        self.FINAL_ANSWER_GENERATOR_INPUT_PATH = os.path.join(PROJECT_ROOT, self.FINAL_ANSWER_GENERATOR_INPUT_PATH)
        self.FINAL_ANSWER_GENERATOR_CORPUS_PATH = os.path.join(PROJECT_ROOT, self.FINAL_ANSWER_GENERATOR_CORPUS_PATH)

        self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH)
        self.FINAL_ANSWER_GENERATOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, self.FINAL_ANSWER_GENERATOR_PROMPT_PATH)

        self.FINAL_ANSWER_GENERATOR_STOP_WORDS = os.getenv("FINAL_ANSWER_GENERATOR_STOP_WORDS", None)
        self.FINAL_ANSWER_GENERATOR_MAX_NEW_TOKENS = os.getenv("FINAL_ANSWER_GENERATOR_MAX_NEW_TOKENS", None)
        self.FINAL_ANSWER_GENERATOR_NUM_WORKERS = int(os.getenv("FINAL_ANSWER_GENERATOR_NUM_WORKERS", 4))
        self.FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES = int(os.getenv("FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES", 300))

        if os.path.exists(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH):
            self.inputs = load_json(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH)
            print(f"Loaded final answer generator {len(self.inputs)} examples from {os.path.relpath(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.")
        else:
            self.inputs = load_json(self.FINAL_ANSWER_GENERATOR_INPUT_PATH)
            print(f"Loaded final answer generator {len(self.inputs)} examples from {os.path.relpath(self.FINAL_ANSWER_GENERATOR_INPUT_PATH, PROJECT_ROOT)}.")
        
        # generator corpus
        self.corpus_data = load_json(self.FINAL_ANSWER_GENERATOR_CORPUS_PATH)
        print(f"Loaded corpus with {len(self.corpus_data)} examples from {os.path.relpath(self.FINAL_ANSWER_GENERATOR_CORPUS_PATH, PROJECT_ROOT)}.")

        # load prompt template
        self.prompt_template = read_text_file(self.FINAL_ANSWER_GENERATOR_PROMPT_PATH)
        print(f"Loaded prompt template from {os.path.relpath(self.FINAL_ANSWER_GENERATOR_PROMPT_PATH, PROJECT_ROOT)}.")

        if self.FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES == -1:
            self.FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES = len(self.inputs)

        self.save_interval = save_interval

    def process_input_content(self, cur_input, cur_prompt):
        try:
            cur_response, prompt_tokens, completion_tokens, _ = call_api_qwen(cur_prompt, TEMPERATURE)
            
            # Thread-safe token accumulation
            with self.token_lock:
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
            
            answers = extract_answers(cur_response)
            cur_input['positive'] = answers['short-answer']['answer']
            cur_input['corrected-answer'] = answers
            return cur_input
        except Exception as e:
            print(f"An error occurred while processing input: {e}")
            return None

    def run(self):
        corpusid_2_context = {cur_dict['id']: cur_dict['context'] for cur_dict in self.corpus_data}

        success_num, all_num = 0, 0
        futures_to_data = {}
        with ThreadPoolExecutor(max_workers=self.FINAL_ANSWER_GENERATOR_NUM_WORKERS) as executor:
            if self.FINAL_ANSWER_GENERATOR_GENERATED_TYPE in ['content']:
                data_list = self.inputs[:self.FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES]
                for data_item in data_list:
                    if 'proposed-questions' not in data_item:
                        continue
                    proposed_questions = data_item['proposed-questions']
                    chunk_id = data_item['id']  # admission.stanford.edu.filter_index.htm.md_chunk_0

                    all_clueid2docid2senidlist = {}
                    objective_facts = data_item['objective-facts']
                    sens = data_item["sens"]
                    for (fact_id, objective_fact), sen in zip(enumerate(objective_facts, start=1), sens):
                        sen_ids = re.findall(r'\d+-\d+|\d+', sen)
                        sen_ids = expand_numbers_and_ranges(sen_ids)
                        all_clueid2docid2senidlist[fact_id] = {
                            chunk_id: sen_ids
                        }

                    for proposed_question_type, proposed_question_dict in proposed_questions.items():
                        if "positive" in proposed_question_dict:
                            continue
                        # get answer with already replaced clues
                        original_question = proposed_question_dict['question']
                        positive_answer = proposed_question_dict['answer']
                        if not positive_answer:
                            continue
                        
                        # print("all_clueid2docid2senidlist:", all_clueid2docid2senidlist)
                        # print("positive_answer:", positive_answer)
                        positive_answer = replace_clue_with_doc_and_sen(all_clueid2docid2senidlist, positive_answer)

                        needed_corpusid2corpus = {chunk_id: corpusid_2_context[chunk_id]}
                        needed_corpusid2corpus_str = list_to_docided_string(needed_corpusid2corpus)
                        
                        cur_prompt = self.prompt_template.replace('[[QUESTION]]', original_question)
                        cur_prompt = cur_prompt.replace('[[CONTEXT]]', needed_corpusid2corpus_str)
                        cur_prompt = cur_prompt.replace('[[ANSWER]]', positive_answer)
                        future = executor.submit(self.process_input_content, proposed_question_dict, cur_prompt)
                        futures_to_data[future] = (
                            None
                        )
                        # futures_to_data[future] = (
                        #     proposed_question_dict.get('rephrased-questions', []),
                        #     proposed_question_dict.get('rephrased-questions-part', []),
                        #     proposed_question_dict.get('rephrased-questions-hybrid', [])
                        # )

                        # rephrased_question_type_list = ['rephrased-questions', 'rephrased-questions-part', 'rephrased-questions-hybrid']
                        # for rephrased_question_type in rephrased_question_type_list:
                        #     rephrased_questions = proposed_question_dict.get(rephrased_question_type, [])
                        #     for rephrased_question_dict in rephrased_questions:
                        #         # get answer with already replaced clues
                        #         if 'reordered-question' in rephrased_question_dict:
                        #             rephrased_question = rephrased_question_dict['reordered-question']
                        #         else:
                        #             rephrased_question = rephrased_question_dict['result']
                        #         positive_answer = rephrased_question_dict['answer']
                        #         positive_answer = replace_clue_with_doc_and_sen(all_clueid2docid2senidlist, positive_answer)
                                
                        #         cur_prompt = self.prompt_template.replace('[[QUESTION]]', rephrased_question)
                        #         cur_prompt = cur_prompt.replace('[[CONTEXT]]', needed_corpusid2corpus_str)
                        #         cur_prompt = cur_prompt.replace('[[ANSWER]]', positive_answer)
                        #         future = executor.submit(self.process_input_content, rephrased_question_dict, cur_prompt)
                        #         futures_to_data[future] = (
                        #             proposed_question_dict.get('rephrased-questions', []),
                        #             proposed_question_dict.get('rephrased-questions-part', []),
                        #             proposed_question_dict.get('rephrased-questions-hybrid', [])
                        #         )

            elif self.FINAL_ANSWER_GENERATOR_GENERATED_TYPE in ['entity_graph']:
                data_list = list(self.inputs.values())[:self.FINAL_ANSWER_GENERATOR_MAX_GEN_TIMES]
                for data_item in data_list:
                    if 'proposed-questions' not in data_item:
                        continue
                    proposed_questions = data_item['proposed-questions']
                    objective_relationships = data_item['selected-relationships']['objective-relationships']
                    objective_relationship_id_2_objective_relationship = {idx: fact for idx, fact in enumerate(objective_relationships, start=1)}
                    
                    all_clueid2docid2senidlist = {}
                    for (relationship_id, objective_relationship_dict) in enumerate(objective_relationships, start=1):
                        docid = objective_relationship_dict['id']
                        sen_ids = re.findall(r'\d+-\d+|\d+', objective_relationship_dict["sentences_used"])
                        sen_ids = expand_numbers_and_ranges(sen_ids)
                        all_clueid2docid2senidlist[relationship_id] = {
                            docid: sen_ids
                        }
                    for proposed_question_type, proposed_question_dict in proposed_questions.items():
                        if "positive" in proposed_question_dict:
                            continue
                        # get answer with already replaced clues
                        original_question = proposed_question_dict['question']
                        positive_answer = proposed_question_dict['answer']
                        # print("all_clueid2docid2senidlist:", all_clueid2docid2senidlist)
                        # print("positive_answer:", positive_answer)
                        positive_answer = replace_clue_with_doc_and_sen(all_clueid2docid2senidlist, positive_answer)

                        # get needed chunk ids
                        needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-relationship-id'])
                        needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
                        needed_related_relationships = [objective_relationship_id_2_objective_relationship[int(clue_id)] for clue_id in needed_objective_relationship_ids if clue_id and int(clue_id) in objective_relationship_id_2_objective_relationship]
                        needed_corpusids = []
                        for relationship_id in needed_objective_relationship_ids:
                            if relationship_id in all_clueid2docid2senidlist:
                                needed_corpusids.extend(list(all_clueid2docid2senidlist[relationship_id].keys()))
                        needed_corpusids = list(sorted(list(set(needed_corpusids))))
                        needed_corpusid2corpus = {
                            cur_related_relationship['id']: corpusid_2_context[cur_related_relationship['id']]
                            for cur_related_relationship in needed_related_relationships if 'id' in cur_related_relationship and cur_related_relationship['id'] in corpusid_2_context
                        } 

                        needed_corpusid2corpus_str = list_to_docided_string(needed_corpusid2corpus)
                        cur_prompt = self.prompt_template.replace('[[QUESTION]]', original_question)
                        cur_prompt = cur_prompt.replace('[[CONTEXT]]', needed_corpusid2corpus_str)
                        cur_prompt = cur_prompt.replace('[[ANSWER]]', positive_answer)
                        future = executor.submit(self.process_input_content, proposed_question_dict, cur_prompt)
                        futures_to_data[future] = (
                            None
                        )
                        # futures_to_data[future] = (
                        #     proposed_question_dict.get('rephrased-questions', []),
                        #     proposed_question_dict.get('rephrased-questions-part', []),
                        #     proposed_question_dict.get('rephrased-questions-hybrid', [])
                        # )

                        # rephrased_question_type_list = ['rephrased-questions', 'rephrased-questions-part', 'rephrased-questions-hybrid']
                        # for rephrased_question_type in rephrased_question_type_list:
                        #     rephrased_questions = proposed_question_dict.get(rephrased_question_type, [])
                        #     for rephrased_question_dict in rephrased_questions:
                        #         # get answer with already replaced clues
                        #         if 'reordered-question' in rephrased_question_dict:
                        #             rephrased_question = rephrased_question_dict['reordered-question']
                        #         else:
                        #             rephrased_question = rephrased_question_dict['result']
                        #         positive_answer = rephrased_question_dict['answer']
                        #         positive_answer = replace_clue_with_doc_and_sen(all_clueid2docid2senidlist, positive_answer)

                        #         cur_prompt = self.prompt_template.replace('[[QUESTION]]', rephrased_question)
                        #         cur_prompt = cur_prompt.replace('[[CONTEXT]]', needed_corpusid2corpus_str)
                        #         cur_prompt = cur_prompt.replace('[[ANSWER]]', positive_answer)
                        #         future = executor.submit(self.process_input_content, rephrased_question_dict, cur_prompt)
                        #         futures_to_data[future] = (
                        #             proposed_question_dict.get('rephrased-questions', []),
                        #             proposed_question_dict.get('rephrased-questions-part', []),
                        #             proposed_question_dict.get('rephrased-questions-hybrid', [])
                        #         )
                               
            else:
                raise ValueError(f"Unknown data file: {self.FINAL_ANSWER_GENERATOR_GENERATED_TYPE}")

            all_num = len(futures_to_data)
            for future in tqdm(as_completed(futures_to_data), total=all_num, desc="Processing Future", dynamic_ncols=True):
                # rephrased_questions, rephrased_questions_part, rephrased_questions_hybrid = futures_to_data[future]
                _ = futures_to_data[future]
                try:
                    cur_response = future.result(timeout=10*60)

                    if cur_response is not None:
                        success_num += 1
                        if (success_num + 1) % self.save_interval == 0:
                            print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.')
                            save_json(self.inputs, self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH)
                except Exception as e:
                    print(f"Error processing future: {e}")

        if success_num or not os.path.exists(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH):
            print(f'Saving outputs to {os.path.relpath(self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH, PROJECT_ROOT)}.')
            save_json(self.inputs, self.FINAL_ANSWER_GENERATOR_OUTPUT_PATH)

        print(f"Total prompt tokens: {self.total_prompt_tokens}")
        print(f"Total completion tokens: {self.total_completion_tokens}")
        print(f"Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%" if all_num > 0 else "Success rate: N/A")

        return self.total_prompt_tokens, self.total_completion_tokens, success_num, all_num