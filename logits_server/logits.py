import torch
import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
import queue
from typing import List

app = FastAPI(
    title="Logits Server",
    description="FastAPI service for calculating token probabilities using Qwen2.5-7B model",
    version="1.0.0",
    contact={
        "name": "Jingjie",
        "email": "zenkit@wiseai.tech"
    }
)

# Create a global request queue
request_queue = queue.Queue()

# Create a thread lock
processing_lock = threading.Lock()

class ProbabilityRequest(BaseModel):
    prompts: List[str]
    answers: List[str]

def calculate_token_probabilities(prompts, answers):
    # Process each prompt and answer individually
    probabilities = []
    answer_input_idss = []
    for prompt, answer in zip(prompts, answers):
        prob, ids = calculate_token_probabilities_single(prompt, answer)
        probabilities.append(prob)
        answer_input_idss.append(ids)
    return probabilities, answer_input_idss

def calculate_token_probabilities_batch(prompts, answers):
    # Tokenize prompts and answers with padding and truncation
    prompt_encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    answer_encodings = tokenizer(answers, return_tensors="pt", padding=True, truncation=True)
    
    prompt_input_ids = prompt_encodings['input_ids']
    answer_input_ids = answer_encodings['input_ids']
    prompt_attention_mask = prompt_encodings['attention_mask']
    answer_attention_mask = answer_encodings['attention_mask']
    
    prompt_lengths = prompt_attention_mask.sum(dim=1)
    answer_lengths = answer_attention_mask.sum(dim=1)
    
    # Combine prompts and answers
    combined_input_ids = []
    combined_attention_masks = []
    
    for i in range(prompt_input_ids.size(0)):
        prompt_tokens = prompt_input_ids[i, :prompt_lengths[i]]
        answer_tokens = answer_input_ids[i, :answer_lengths[i]]
        combined_tokens = torch.cat([prompt_tokens, answer_tokens], dim=0)
        combined_input_ids.append(combined_tokens)
        
        prompt_mask = prompt_attention_mask[i, :prompt_lengths[i]]
        answer_mask = answer_attention_mask[i, :answer_lengths[i]]
        combined_mask = torch.cat([prompt_mask, answer_mask], dim=0)
        combined_attention_masks.append(combined_mask)
    
    # Pad sequences
    combined_input_ids = torch.nn.utils.rnn.pad_sequence(
        combined_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    combined_attention_masks = torch.nn.utils.rnn.pad_sequence(
        combined_attention_masks, batch_first=True, padding_value=0
    )
    
    inputs = {
        'input_ids': combined_input_ids.to(model.device),
        'attention_mask': combined_attention_masks.to(model.device)
    }
    
    labels = combined_input_ids.clone().to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
    
    # Extract probabilities for the answer portion
    answer_token_probs = []
    answer_input_idss = []
    
    for i in range(len(prompts)):
        prompt_len = prompt_lengths[i].item()
        answer_len = answer_lengths[i].item()
        
        # Get the answer token IDs and probabilities
        answer_input_ids_seq = combined_input_ids[i, prompt_len:prompt_len + answer_len].to(model.device)
        answer_probs = probs[i, prompt_len - 1:prompt_len + answer_len - 1].to(model.device)
        
        # Collect probabilities of the actual answer tokens
        token_probs = answer_probs.gather(1, answer_input_ids_seq.unsqueeze(1)).squeeze(-1)
        answer_token_probs.append(token_probs.tolist())
        answer_input_idss.append(answer_input_ids_seq.tolist())
    
    return answer_token_probs, answer_input_idss

def calculate_token_probabilities_single(prompt, answer):
    # Tokenize prompt and answer individually
    prompt_encoding = tokenizer(prompt, return_tensors="pt", truncation=True)
    answer_encoding = tokenizer(answer, return_tensors="pt", truncation=True)
    
    prompt_input_ids = prompt_encoding['input_ids'][0]
    answer_input_ids = answer_encoding['input_ids'][0]
    prompt_attention_mask = prompt_encoding['attention_mask'][0]
    answer_attention_mask = answer_encoding['attention_mask'][0]
    
    prompt_len = prompt_attention_mask.sum().item()
    answer_len = answer_attention_mask.sum().item()
    
    # Combine prompt and answer
    combined_input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=0)
    combined_attention_mask = torch.cat([prompt_attention_mask, answer_attention_mask], dim=0)
    
    inputs = {
        'input_ids': combined_input_ids.unsqueeze(0).to(model.device),
        'attention_mask': combined_attention_mask.unsqueeze(0).to(model.device)
    }
    
    labels = combined_input_ids.unsqueeze(0).clone().to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
    
    # Get the answer token IDs and probabilities
    answer_input_ids_seq = combined_input_ids[prompt_len:prompt_len + answer_len].to(model.device)
    answer_probs = probs[prompt_len - 1:prompt_len + answer_len - 1].to(model.device)
    
    # Collect probabilities of the actual answer tokens
    token_probs = answer_probs.gather(1, answer_input_ids_seq.unsqueeze(1)).squeeze(-1)
    
    return token_probs.tolist(), answer_input_ids_seq.tolist()

def process_requests():
    while True:
        # Retrieve a request from the queue
        item = request_queue.get()
        if item == None:
            break  # Termination signal received, exit loop
        
        data, res = item  # 'data' is the extracted JSON data
        
        with processing_lock:
            prompts = data.get('prompts')
            answers = data.get('answers')
            
            if not prompts or not answers:
                response_data = {"error": "Missing prompts or answers"}
                status_code = 400
            else:
                probabilities, answer_input_idss = calculate_token_probabilities(prompts, answers)
                response_data = {"probabilities": probabilities, "answer_input_ids": answer_input_idss}
                status_code = 200
            
            # Place the result in the response queue
            res.put((response_data, status_code))
        
        request_queue.task_done()

# Start the request processing thread
processing_thread = threading.Thread(target=process_requests, daemon=True)
processing_thread.start()

@app.post('/calculate_probabilities')
def calculate_probabilities(request: ProbabilityRequest):
    # Extract data from the request
    data = request.model_dump()

    # Create a response queue for each request
    response_queue = queue.Queue()

    # Place the data and response queue into the global request queue
    request_queue.put((data, response_queue))

    # Wait for processing to complete and get the response
    response_data, status_code = response_queue.get()

    return response_data

if __name__ == '__main__':
    model_path = os.getenv("MODEL_PATH", None)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, fix_mistral_regex=True)

    # Load the model and implement model parallelism
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map='auto',  # Automatically allocate devices for model parallelism
        dtype=torch.float16  # Use half precision to speed up inference and reduce memory usage
    )

    # Start the FastAPI application
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)