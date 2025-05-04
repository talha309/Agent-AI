# Temperature controls the randomness of LLM output. Lower values (e.g., 0.2) make outputs more deterministic and focused, ideal for factual or technical responses.
# Higher values (e.g., 1.0+) increase creativity but may reduce coherence. Default is often 1.0 for balanced output.
temperature = 0.7  # Set to balance coherence and diversity for this task.
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
result = llm.invoke(input("Enter your Prompt: "))
print(result)

