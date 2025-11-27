
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

def call_api(query, temperature=0):
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
        temperature = 1
    )
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    content = completion.choices[0].message.content
    return content,prompt_tokens,completion_tokens,temperature


if __name__ == "__main__":
    print(call_api_qwen("testing, testing 1,2,3",1))
    print(call_api("testing, testing 1,2,3",1))