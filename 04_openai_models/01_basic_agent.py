from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_KEY = os.getenv('GOOGLE_API_KEY')

client = AsyncOpenAI(
    api_key=GOOGLE_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


agent = Agent(
    name="Assistant",
    instructions="You are an expert of agentic AI.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

query = input("Enter the query: ")

result = Runner.run_sync(
    agent,
    query,
)

print(result.final_output)