import html
import json
import os
import re
import unicodedata

import gradio as gr
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
    description="Multi-label toxicity classifier (DistilBERT, fine-tuned) - PROG74040 Group 2",
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


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return predict([req.text])[0]


@app.post("/batch")
def batch(req: BatchRequest):
    return {"results": predict(req.texts)}


def gradio_predict(text):
    result = predict([text])[0]
    flagged = [label for label, is_flagged in result["flagged"].items() if is_flagged]
    flagged_text = ", ".join(flagged) if flagged else "none"
    return result["scores"], flagged_text


demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(lines=4, placeholder="Enter a comment to analyze...", label="Comment"),
    outputs=[
        gr.Label(label="Per-category scores"),
        gr.Textbox(label="Flagged categories"),
    ],
    title="Toxicity Detection (DistilBERT)",
    description=(
        "Multi-label toxicity classifier fine-tuned on the Jigsaw Toxic Comment dataset "
        "(PROG74040, Group 2). Detects: toxic, severe_toxic, obscene, threat, insult, "
        "identity_hate. A REST API is also available at /analyze and /batch (see /docs)."
    ),
    examples=[
        ["Thanks for the helpful edit, this looks much better now."],
        ["You are a childish, pompous, self-regarding fool."],
    ],
)

app = gr.mount_gradio_app(app, demo, path="/")
