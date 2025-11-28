from dotenv import load_dotenv
from openai import OpenAI
import os
import numpy as np

load_dotenv()

def call_api_simple(query, temperature=0):
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": query}],
        temperature = 1
    )
    return completion.choices[0].message.content 


def call_api_qwen(query, temperature=0):
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": query}],
        temperature=temperature
    )
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    content = completion.choices[0].message.content
    return content,prompt_tokens,completion_tokens,temperature


def get_qwen_embeddings(texts, dim=1024):
    # Refference for embeddings model:
    # https://www.alibabacloud.com/help/en/model-studio/embedding?spm=a2c63.l28256.help-menu-2400256.d_0_8_0.95cf4f2cBwSJ26
    # dim options = 2,048, 1,536, 1,024 (default), 768, 512, 256, 128, 64
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=texts,
        dimensions = dim,
    )

    embeddings = [np.array(data.embedding) for data in response.data]
    total_tokens = response.usage.total_tokens

    return embeddings, total_tokens


if __name__ == "__main__":
    pass