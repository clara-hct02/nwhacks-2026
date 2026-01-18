from fastapi import FastAPI
import torch
from google import genai
from Server.schemas import MessageRequest, PredictionResponse, ReasoningResponse

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: MessageRequest):
    message = req.message

    threat = "RED"
    return {"threatLevel": threat}

@app.post("/reason", response_model=ReasoningResponse)
async def reason(req: MessageRequest):
    message = req.message

    # Foundation Step: Send to Gemini for Risk Assessment
    prompt = (
        "This message has been flagged for possible threats.\n"
        f'Message: "{message}"\n\n'
        "Return a one sentence reason as to why this message may be a threat."
    )

    response = genai.GenerativeModel("gemini-3-flash-preview").generate_content(prompt)

    return {"reasoning": response.text} 

