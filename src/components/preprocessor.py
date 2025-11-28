import re
import os
import json
from pathlib import Path
from ..tools.api import call_api_qwen
from ..tools.json_utils import save_json, load_json
from ..tools.string_utils import extract_text_from_pdf, read_text_file
from .. import PROJECT_ROOT


def split_by_questions(text):
    """
    Split into blocks using Q markers like "1)", "2)", ..., "10)"
    """
    pattern = r"(?m)^(?:\d{1,2}\))"
    matches = list(re.finditer(pattern, text))

    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        chunks.append(chunk)

    return chunks


class Preprocessor:
    def __init__(self, verbose):
        self.verbose = verbose
        self.PREPROCESSOR_PDF_PATH = None
        self.PREPROCESSOR_PROMPT_PATH = None
        self.PREPROCESSOR_CLEANED_OUTPUT_PATH = None
        self.PREPROCESSOR_CHUNKED_OUTPUT_PATH = None

        if os.getenv("PREPROCESSOR_PDF_PATH", None) != None:
            self.PREPROCESSOR_PDF_PATH = os.getenv("PREPROCESSOR_PDF_PATH")
            self.PREPROCESSOR_PROMPT_PATH = os.getenv("PREPROCESSOR_PROMPT_PATH")
            self.PREPROCESSOR_CLEANED_OUTPUT_PATH = os.getenv("PREPROCESSOR_CLEANED_OUTPUT_PATH")
            self.PREPROCESSOR_CHUNKED_OUTPUT_PATH = os.getenv("PREPROCESSOR_CHUNKED_OUTPUT_PATH")
            self.PREPROCESSOR_CLEANED_PDF_PATH = os.getenv("PREPROCESSOR_CLEANED_PDF_PATH")
        else:
            raise EnvironmentError("Environment variables are not defined correctly")

        self.PREPROCESSOR_PDF_PATH = os.path.join(PROJECT_ROOT, self.PREPROCESSOR_PDF_PATH)
        self.PREPROCESSOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, self.PREPROCESSOR_PROMPT_PATH)
        self.PREPROCESSOR_CLEANED_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.PREPROCESSOR_CLEANED_OUTPUT_PATH)
        self.PREPROCESSOR_CHUNKED_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.PREPROCESSOR_CHUNKED_OUTPUT_PATH)

    def process_chunk_text(self, chunk, counter):
        """
        Process chunk text and add sentence labels
        """
        lines = []
        for line in chunk.split('\n'):
            sentences = re.split(r'(?<=[.!?])\s+', line)
            new_line = []

            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                # Tag each sentence
                if sent[-1] in ".!?":
                    new_line.append(f"{sent[:-1]} [Sen {counter}]{sent[-1]}")
                else:
                    new_line.append(f"{sent} [Sen {counter}]")

                counter += 1

            lines.append(" ".join(new_line))

        return "\n".join(lines), counter

    def extract_question_from_chunk(self, question):
        """
        Extract the question text from a chunk
        """
        match = re.match(r"(\d+\))\s*(.*?)([?.])", question)
        if match:
            q_number = match.group(1)
            q_text = match.group(2).strip()
            punctuation = match.group(3)
            question_only = f"{q_number} {q_text}{punctuation}"
            answer_body = question[len(match.group(0)):].strip()
        else:
            question_only = ""
            answer_body = question

        return question_only, answer_body

    def process_questions(self, cleaned_questions):
        """
        Process questions and add sentence labels
        """
        chunk_contents = []
        counter = 1
        question_no = 1

        for question in cleaned_questions:
            question_only, answer_body = self.extract_question_from_chunk(question)
            q_processed, counter = self.process_chunk_text(answer_body, counter)

            chunk_contents.append({
                "id": f"doc1_question{question_no}",
                "question": question_only,
                "origin_context": question,
                "context": q_processed
            })

            question_no += 1

        return chunk_contents

    def run(self):
        if self.verbose:
            print("Loading and cleaning PDF......")
        data_cleaning_prompt = read_text_file(self.PREPROCESSOR_PROMPT_PATH)
        # Extract text from pdf, call llm to clean up the extracted text
        pdf_text = extract_text_from_pdf(self.PREPROCESSOR_PDF_PATH)
        cleaned_pdf_text,prompt_tokens,completion_tokens,_ = call_api_qwen(data_cleaning_prompt + pdf_text)
        save_json(cleaned_pdf_text, self.PREPROCESSOR_CLEANED_PDF_PATH)

        # Split by question 1) 2)
        if self.verbose:
            print("Splitting Questions......")
        question_set = split_by_questions(cleaned_pdf_text)
        save_json(question_set, self.PREPROCESSOR_CLEANED_OUTPUT_PATH)

        # chunking questions
        cleaned_questions = load_json(self.PREPROCESSOR_CLEANED_OUTPUT_PATH)
        if self.verbose:
            print("Chunking questions......")
        chunk_contents = self.process_questions(cleaned_questions)
        save_json(chunk_contents, self.PREPROCESSOR_CHUNKED_OUTPUT_PATH)

        return prompt_tokens, completion_tokens


if __name__ == "__main__":
    pass