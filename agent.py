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
system = "Use a concise narrative style when describing something.  Do not repeat text already given unless asked."
messages = []

while True:
    input_text = input("You: ")

    if (input_text.strip().lower() == "quit"):
        break

    if (input_text == ""):
        continue

    messages.append(create_user_message(input_text))

    response = run_turn(messages, system)

    output_text = "".join(block.text for block in response.content if block.type == "text")

    print(output_text)

    messages.append(create_assistant_message(content=response.content))

# %%
