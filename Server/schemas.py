from pydantic import BaseModel

class MessageRequest(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    threatLevel: str

class ReasoningResponse(BaseModel):
    reasoning: str