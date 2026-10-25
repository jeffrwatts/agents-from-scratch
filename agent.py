# %%
from dotenv import load_dotenv
import anthropic
import json
from datetime import datetime
from anthropic.types import ToolParam

load_dotenv()
client = anthropic.Anthropic()
model = "claude-opus-5-5"
max_tokens = 10000

# %% Messages
def create_user_message(text):
    content = {"type": "text", "text": text}
    message = {"role": "user", "content": [content]}
    return message

def create_assistant_message(content):
    blocks = [block.model_dump(exclude_none=True) for block in content]
    message = {"role": "assistant", "content": blocks}
    return message

# %% Tools
def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)

get_current_datetime_schema = ToolParam({
    "name": "get_current_datetime",
    "description": "Returns the current date and time formatted according to the specified format",
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": "A string specifying the format of the returned datetime. Uses Python's strftime format codes.",
                "default": "%Y-%m-%d %H:%M:%S"
            }
        },
        "required": []
    }
})

tool_functions = {"get_current_datetime": get_current_datetime}

def run_tools(content):
    results = []
    for block in content:
        if block.type != "tool_use":
            continue
        print(f"  [tool] {block.name}({block.input})")
        try:
            output = tool_functions[block.name](**block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        except Exception as e:
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": f"Error: {e}", "is_error": True})
    return {"role": "user", "content": results}

# %% turn
def run_turn(messages, system=None):
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools":[get_current_datetime_schema],
    }
    
    if system:
        params["system"] = system
    
    return client.messages.create(**params)


# %% Agentic Loop
system = "Use a concise narrative style when describing something.  Do not repeat text already given unless asked."
messages = []

while True:
    input_text = input("You: ")

    if (input_text.strip().lower() == "quit"):
        break

    if (input_text.strip() == ""):
        continue

    messages.append(create_user_message(input_text))

    response = run_turn(messages, system)

    while response.stop_reason == "tool_use":
        messages.append(create_assistant_message(content=response.content))
        messages.append(run_tools(response.content))
        response = run_turn(messages, system)

    output_text = "".join(block.text for block in response.content if block.type == "text")
    print(output_text)

    messages.append(create_assistant_message(content=response.content))


# %%
