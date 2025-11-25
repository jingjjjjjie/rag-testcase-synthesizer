import re
from ..tools.api import call_api
from ..tools.json_utils import save_json, load_json
from ..tools.string_utils import extract_text_from_pdf, read_text_file


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

def process_chunk_text(chunk, counter):
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


if __name__ == "__main__":
    save_cleaned_questions_path = "src/data/cleaned_questions.json"
    save_chunked_questions_path = "src/data/chunked_questions.json"
    data_cleaning_prompt_path = "src/prompts/data_cleaning.txt"
    pdf_path = "src/files/sample_data.pdf"

    # read the data cleaning prompt
    data_cleaning_prompt = read_text_file(data_cleaning_prompt_path)

    # extract text from pdf, call llm to clean up the the extracted text
    pdf_text = extract_text_from_pdf(pdf_path)
    cleaned_pdf_text = call_api(data_cleaning_prompt + pdf_text)

    # split by question 1) 2)
    question_set = split_by_questions(cleaned_pdf_text)

    # save the intermediate result
    save_json(question_set, save_cleaned_questions_path)
    cleaned_questions = load_json(save_cleaned_questions_path)

    # loop through the cleaned_questions, add [sen x] labels
    counter = 1
    question_no = 1
    chunk_contents = []

    for question in cleaned_questions:

    # 1. Extract the question text
        match = re.match(r"(\d+\))\s*(.*?)(\?)", question)
        if match:
            q_number = match.group(1)               # "1)"
            q_text = match.group(2).strip() + "?"   # "Can workers opt out of the savings program?"
            question_only = f"{q_number} {q_text}"
            
            # Remove the extracted question part from the full chunk
            answer_body = question[len(match.group(0)):].strip()
        else:
            # fallback: no question format detected
            question_only = ""
            answer_body = question

        # 2. Process only the answer body for Sen tagging
        q_processed, counter = process_chunk_text(answer_body, counter)

        chunk_contents.append(
            {
                "id": f"doc1_question{question_no}",
                "question": question_only,
                "origin_context": question,
                "context": q_processed
            }
        )

        question_no += 1

    save_json(chunk_contents, save_chunked_questions_path)
    chunk_contents = load_json(save_chunked_questions_path)
    print(chunk_contents)