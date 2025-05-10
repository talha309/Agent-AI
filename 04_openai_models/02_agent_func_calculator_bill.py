from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner,function_tool
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_KEY = os.getenv('GOOGLE_API_KEY')

client = AsyncOpenAI(
    api_key=GOOGLE_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

@function_tool
def calculate_bill(units: float) -> float:
    """
    Calculates electricity bill and prints the amount.
    """
    try:
        price_per_unit = 0.5
        bill = units * price_per_unit
        print(f"Total units: {units}, Price per unit: {price_per_unit}, Bill: {bill}")
        return bill
    except Exception as e:
        print("Error occurred:", str(e))
        return str(e)

agent = Agent(
    name="Assistant",
    instructions="You are an expert of agentic AI.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    tools=[calculate_bill],
)

query = input("Enter the query: ")

result = Runner.run_sync(
    agent,
    query,
)

print(result.final_output)