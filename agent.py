# %%
from dotenv import load_dotenv
import anthropic
import json

load_dotenv()
client = anthropic.Anthropic()
model = "claude-opus-5-5"
max_tokens = 10000

# %%
def create_user_message(text):
    content = {"type": "text", "text": text}
    message = {"role": "user", "content": [content]}
    return message

def create_assistant_message(content):
    blocks = [block.model_dump(exclude_none=True) for block in content]
    message = {"role": "assistant", "content": blocks}
    return message

def run_turn(messages, system=None):
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    
    if system:
        params["system"] = system
    
    return client.messages.create(**params)


# %%
system = None # "respond like a pirate"
messages = []

messages.append(create_user_message(text="What is Quantum Computing.  Answer in one sentence."))

response = run_turn(messages, system)

messages.append(create_assistant_message(content=response.content))

messages.append(create_user_message(text="add another sentence to expand further"))

response = run_turn(messages, system)

messages.append(create_assistant_message(content=response.content))

# %%
print(json.dumps(messages, indent=2))


# %%
