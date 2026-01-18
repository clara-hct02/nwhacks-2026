from pydantic import BaseModel

class MessageRequest(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    threatLevel: str
    reasoning: str
