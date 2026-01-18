from fastapi import FastAPI
import torch
from google import genai
from Server.schemas import MessageRequest, PredictionResponse
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

app = FastAPI()
client = genai.Client(api_key=GEMINI_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000"],  # or ["*"] for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/analyze", response_model=PredictionResponse)
async def analyze(req: MessageRequest):
    message = req.message

    # The extension already knows the threat level.
    # Backend ONLY generates the AI reason.
    reason = get_reason(message)

    return {"threatLevel": "RED", "reasoning": reason}
   


def get_reason(message):
    prompt = (
        "This message has been flagged for possible threats.\n"
        f'Message: "{message}"\n\n'
        "Return a one sentence reason as to why this message may be a threat."
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )
    return response.text

