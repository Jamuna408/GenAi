import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv(override=True)

# Read API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. Check your .env file."
    )

# Create OpenAI client
client = OpenAI(api_key=api_key)

# Send prompt to the model
response = client.responses.create(
    model="gpt-5.5",
    input="What is Artificial Intelligence? Explain in simple words."
)

# Display response
print("\nLLM Response:")
print(response.output_text)