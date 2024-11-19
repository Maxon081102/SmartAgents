from openai import OpenAI
import json

# Set up API client with the vLLM server settings
openai_api_key = ""
openai_api_base = "http://localhost:8000/v1/"
client = OpenAI(api_key="sfe", base_url=openai_api_base)

def query_historical_event(date):
    # Dictionary of historical events keyed by date
    historical_events = {
        "1969-07-20": "On July 20, 1969, Neil Armstrong and Buzz Aldrin became the first humans to land on the Moon during NASA's Apollo 11 mission.",
        "1989-11-09": "On November 9, 1989, the Berlin Wall fell, marking the symbolic end of the Cold War and leading to the reunification of East and West Germany.",
        "2001-09-11": "On September 11, 2001, terrorist attacks in the United States resulted in the destruction of the World Trade Center in New York City and significant loss of life.",
        # Add more dates and events as needed
    }

    # Return the event if the date exists in the dictionary, otherwise a default message
    return historical_events.get(date, f"No historical event information available for {date}.")


# Define the tool for historical events
tools = [
    {
        "function": {
            "name": "query_historical_event",
            "description": "Provides information about a historical event that occurred on a specified date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The date of the event in YYYY-MM-DD format."
                    },
                },
                "required": ["date"]
            }
        }
    }
]

# Define tool_choice with the function details
tool_choice = {
    "type": "function",
    "function": {
        "name": "query_historical_event"
    }
}

# Define the conversation with function calling capabilities
messages = [
    {"role": "system", "content": "You are a knowledgeable assistant that can retrieve information about historical events."},
    {"role": "user", "content": "Can you tell me what happened on November 9, 1989?"},
]

# Generate a response using the messages and function tool
chat_response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
    temperature=0.7,
    max_tokens=1024,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.6,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "query_historical_event"}}
)

# Check for the function call and handle it
if chat_response.choices[0].message.tool_calls:
    # Extract and parse the date argument from the function call
    date_argument = chat_response.choices[0].message.tool_calls[0].function.arguments
    date_argument_dict = json.loads(date_argument)  # Parse the JSON string to a dictionary
    date = date_argument_dict.get("date", None)

    # Call the query_historical_event function with the date
    response = query_historical_event(date)
    print("Assistant response:", response)
else:
    # If there's no function call, display the assistant's regular response
    print("Assistant response:", chat_response.choices[0].message.content)