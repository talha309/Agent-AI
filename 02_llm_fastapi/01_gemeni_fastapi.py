# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize the GoogleGenerativeAI model using the API key from .env file
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Google API Key is missing!")

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,           # Slightly creative responses
    top_p=0.9,                 # Use nucleus sampling
    top_k=40,                  # Consider top 40 tokens for sampling
    max_output_tokens=512     # Limit output to 512 tokens
)


# Define the request body schema
class PromptRequest(BaseModel):
    prompt: str

# Define the API endpoint that receives the prompt and generates a response
@app.post("/generate")
async def generate_text(request: PromptRequest):
    try:
        # Call the GoogleGenerativeAI to get the result
        result = llm.invoke(request.prompt)

        # Check if the result is valid
        if result is None or result == "":
            raise HTTPException(status_code=500, detail="Received empty or null response from the model")

        # Return the response in JSON format
        return {"response": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

