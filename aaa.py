from dotenv import load_dotenv
import os
from perplexity import Perplexity

load_dotenv("environment_test.env")  # Load environment variables from .env file
client = Perplexity(api_key=os.getenv("PERPLEXITY_API_KEY")) # Uses PERPLEXITY_API_KEY from .env file

completion = client.chat.completions.create(
    model="sonar-pro",
    messages=[
        {"role": "user", "content": "explain what is API that a 10 year old can understand?"}
    ]
)

print(completion.choices[0].message.content)