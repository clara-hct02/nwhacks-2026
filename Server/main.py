from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Server.schemas import MessageRequest, PredictionResponse
from google import genai
from dotenv import load_dotenv
import torch
import os
import json
from .model import SpamClassifier
from .training import SimpleTokenizer  # tokenizer class from your training script

load_dotenv()

# -----------------------------
# Gemini API
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # extension runs locally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Trained Model Checkpoint
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "spam_classifier.pt")

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

word2idx = checkpoint["tokenizer_word2idx"]
config = checkpoint["config"]
best_params = checkpoint["best_params"]
embedding_dim = checkpoint["embedding_dim"]

# Rebuild tokenizer
tokenizer = SimpleTokenizer()
tokenizer.word2idx = word2idx
tokenizer.idx2word = {i: w for w, i in word2idx.items()}

# Rebuild model
model = SpamClassifier(
    vocab_size=len(word2idx),
    embedding_dim=embedding_dim,
    hidden_dim=best_params["hidden_dim"],
    dropout=best_params["dropout"],
    pretrained_embeddings=None,  # embeddings already inside weights
    freeze_embeddings=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Thresholds from training config
MEDIUM_THRESHOLD = config["medium_threshold"]
HIGH_THRESHOLD = config["high_threshold"]
MAX_SEQ_LEN = config["max_seq_length"]


# -----------------------------
# Helper: classify message
# -----------------------------
def classify_message(text: str) -> str:
    encoded = tokenizer.encode(text, MAX_SEQ_LEN)
    x = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    if prob >= HIGH_THRESHOLD:
        return "RED"
    elif prob >= MEDIUM_THRESHOLD:
        return "YELLOW"
    else:
        return "GREEN"


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Watchdog backend running"}


@app.post("/classify")
async def classify(req: MessageRequest):
    threat = classify_message(req.message)
    return {"threatLevel": threat}


@app.post("/reason")
async def reason(req: MessageRequest):
    prompt = (
        "This message has been flagged for possible threats.\n"
        f'Message: "{req.message}"\n\n'
        "Return a one sentence reason as to why this message may be a threat."
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )

    return {"reasoning": response.text}
