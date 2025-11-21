import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # .env を読み込む

FORGE_API_KEY = os.getenv("FORGE_API_KEY")
 
client = OpenAI(
    base_url="https://api.forge.tensorblock.co/v1", 
    api_key=FORGE_API_KEY,  
)
 
models = client.models.list()
print(f"models: {models}")

completion = client.chat.completions.create(
    model="tensorblockI/gpt-4o",
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
 
print(completion.choices[0].message)

# Calculate point-biserial correlation
'''
r = (m1 - m0)\sqrt(p(1-p)) / s_x
'''
rpb_list = []