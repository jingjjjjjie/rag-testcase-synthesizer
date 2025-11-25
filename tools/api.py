
from dotenv import load_dotenv
from openai import OpenAI
import os

def call_api(query, temperature=0):
    load_dotenv()

    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": query}],
        temperature = 1
    )
    return(completion.choices[0].message.content)




if __name__ == "__main__":
    print(call_api("testing, testing 1,2,3",1))