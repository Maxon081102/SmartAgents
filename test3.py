from openai import OpenAI
import json
import traceback

# Set up API client with the vLLM server settings
openai_api_key = ""
openai_api_base = "http://localhost:8000/v1/"
client = OpenAI(api_key="sfe", base_url=openai_api_base)


def code_execute(code):
    try:
        exec(code.replace("\\n", "\n"))
    except Exception as e:
        return traceback.format_exc()
    return "DONE"


def code_check(code, tests):
    try:
        exec("\n\n".join([code.replace("\\n", "\n"), tests]))
    except Exception as e:
        return traceback.format_exc()
    return "DONE"


# Define the tool for historical events
tools = [
    {
        "function": {
            "name": "code_execute",
            "description": "Execute code and return done if successful else return the error message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "code to execute"
                    },
                },
                "required": ["code"]
            }
        }
    },
    {
        "function": {
            "name": "code_check",
            "description": "Execute code with tests and return done if successful else return the error message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "code to execute"
                    },
                    "tests": {
                        "type": "string",
                        "description": "tests to check"
                    },
                },
                "required": ["code", "tests"]
            }
        }
    },
]

# Define tool_choice with the function details
tool_choice = {
    "type": "function",
    "function": {
        "name": "code_execute",
        "name": "code_check",
    }
}

function_mapping = {
    "code_execute": code_execute,
    "code_check": code_check,
}

# Define the conversation with function calling capabilities
messages = [
    {"role": "system", "content": "You are a programmer assistant."},
    {"role": "user", "content": 
"""Complete function 
```python
from typing import List 
def intersperse(numbers: List[int], delimeter: int) -> List[int]: 
    \""" Insert a number 'delimeter' between every two consecutive elements of input list `numbers' 
    >>> intersperse([], 4) [] 
    >>> intersperse([1, 2, 3], 4) [1, 4, 2, 4, 3] 
    \"""
```

and tests for it
```python
assert intersperse([], 7) == [] 
assert intersperse([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2] 
assert intersperse([2, 2, 2], 2) == [2, 2, 2, 2, 2]
```"""},
]

done = True

while done:
    chat_response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.6,
        tools=tools,
        tool_choice=tool_choice
    )

    # Check for the function call and handle it
    if chat_response.choices[0].message.tool_calls:
        # Extract and parse the date argument from the function call
        func = function_mapping[chat_response.choices[0].message.tool_calls[0].function.name]
        date_argument = chat_response.choices[0].message.tool_calls[0].function.arguments
        date_argument_dict = json.loads(date_argument) 

        # Call the query_historical_event function with the date
        print(date_argument_dict)
        response = func(**date_argument_dict)
        print("Assistant response:", response)
        if response == "DONE":
            done = False
        else:
            messages.append({"role": "assistant", "content" : ""})
            messages.append({"role": "user", "content" :"Try more: " + response})
    else:
        # If there's no function call, display the assistant's regular response
        print("Assistant response:", chat_response.choices[0].message.content)