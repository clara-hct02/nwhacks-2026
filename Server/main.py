from fastapi import FastAPI
import torch
from google import genai
from Server.schemas import MessageRequest, PredictionResponse

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/analyze", response_model=PredictionResponse)
async def analyze(req: MessageRequest):
    message = req.message

    threat = "RED"
    # threat = assess_threat(message)

    # reason = "test reason"

    if threat == "RED" or threat == "YELLOW":
        reason = get_reason(message)
    else:
        reason = None

    return {"threatLevel": threat, "reasoning": reason}     


def get_reason(message):
    prompt = (
        "This message has been flagged for possible threats.\n"
        f'Message: "{message}"\n\n'
        "Return a one sentence reason as to why this message may be a threat."
    )

    response = genai.GenerativeModel("gemini-3-flash-preview").generate_content(prompt)
    return response.text

# def assess_threat(message):
#     model()

