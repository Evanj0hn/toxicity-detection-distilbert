import html
import json
import os
import re
import unicodedata

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LENGTH = 256
MODEL_DIR = os.environ.get("MODEL_DIR", "model_artifact")
THRESHOLDS_PATH = os.environ.get("THRESHOLDS_PATH", "best_thresholds.json")

app = FastAPI(
    title="Toxicity Detection API",
    description="Multi-label toxicity classifier (DistilBERT, fine-tuned) — PROG74040 Group 2",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

with open(THRESHOLDS_PATH) as f:
    thresholds = json.load(f)


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"==+.*?==+", " ", text)
    text = re.sub(r"\[\[.*?\]\]", " ", text)
    text = re.sub(r"\{\{.*?\}\}", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(texts: list[str]) -> list[dict]:
    cleaned = [clean_text(t) for t in texts]
    enc = tokenizer(
        cleaned, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()

    results = []
    for row in probs:
        scores = {label: round(float(row[i]), 4) for i, label in enumerate(LABEL_COLS)}
        flagged = {label: bool(row[i] >= thresholds[label]) for i, label in enumerate(LABEL_COLS)}
        results.append({"scores": scores, "flagged": flagged})
    return results


class AnalyzeRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: list[str]


@app.get("/")
def root():
    return {
        "name": "Toxicity Detection API",
        "docs": "/docs",
        "endpoints": ["/analyze", "/batch", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return predict([req.text])[0]


@app.post("/batch")
def batch(req: BatchRequest):
    return {"results": predict(req.texts)}
