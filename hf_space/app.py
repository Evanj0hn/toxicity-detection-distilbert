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


CATEGORY_INFO = {
    "toxic": "General rude, disrespectful, or unreasonable content",
    "severe_toxic": "Extremely hateful, aggressive, or violent language",
    "obscene": "Profane or vulgar language",
    "threat": "An explicit statement of intent to harm someone",
    "insult": "Demeaning or disparaging language directed at a person",
    "identity_hate": "Hateful language targeting identity (race, religion, gender, etc.)",
}


def render_results_html(scores: dict, flagged: dict) -> str:
    any_flagged = any(flagged.values())
    verdict_color = "#dc2626" if any_flagged else "#16a34a"
    verdict_bg = "#fef2f2" if any_flagged else "#f0fdf4"
    verdict_text = "⚠️ Flagged as potentially harmful" if any_flagged else "✅ No toxicity detected"
    flagged_list = ", ".join(l.replace("_", " ") for l, v in flagged.items() if v) or "None"

    rows = []
    for label in LABEL_COLS:
        score = scores[label]
        is_flagged = flagged[label]
        pct = round(score * 100, 1)
        bar_color = "#dc2626" if is_flagged else "#3b82f6"
        badge = (
            '<span style="background:#fee2e2;color:#dc2626;padding:2px 8px;'
            'border-radius:12px;font-size:11px;font-weight:600;margin-left:8px;">FLAGGED</span>'
            if is_flagged else ""
        )
        rows.append(f"""
        <div style="margin-bottom:16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
            <span style="font-weight:600;color:#1f2937;font-size:14px;">{label.replace('_', ' ').title()}{badge}</span>
            <span style="font-weight:700;color:#374151;font-size:14px;">{pct}%</span>
          </div>
          <div style="background:#e5e7eb;border-radius:8px;height:10px;overflow:hidden;">
            <div style="background:{bar_color};height:100%;width:{pct}%;border-radius:8px;"></div>
          </div>
          <div style="font-size:12px;color:#6b7280;margin-top:3px;">{CATEGORY_INFO[label]}</div>
        </div>
        """)

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:4px;">
      <div style="background:{verdict_bg};border:1px solid {verdict_color}44;border-radius:12px;
                  padding:16px 18px;margin-bottom:20px;">
        <div style="font-size:17px;font-weight:700;color:{verdict_color};">{verdict_text}</div>
        <div style="font-size:13px;color:#4b5563;margin-top:4px;">Flagged categories: {flagged_list}</div>
      </div>
      {''.join(rows)}
    </div>
    """


EMPTY_RESULTS_HTML = render_results_html({l: 0.0 for l in LABEL_COLS}, {l: False for l in LABEL_COLS})


def gradio_predict(text: str) -> str:
    if not text or not text.strip():
        return EMPTY_RESULTS_HTML
    result = predict([text])[0]
    return render_results_html(result["scores"], result["flagged"])


CUSTOM_CSS = """
.gradio-container {max-width: 960px !important; margin: auto !important;}
#header-md h1 {margin-bottom: 0px;}
"""

with gr.Blocks(title="Toxicity Detection - DistilBERT") as demo:
    with gr.Column(elem_id="header-md"):
        gr.Markdown(
            """
            # \U0001F6E1️ Toxicity Detection
            #### Multi-label comment classifier &middot; fine-tuned DistilBERT

            Detects six categories of harmful content in a single pass: **toxic**, **severe toxic**,
            **obscene**, **threat**, **insult**, and **identity-based hate**. Built for
            **PROG74040 &mdash; Advanced Topics in AI and ML**, Conestoga College &middot; Group 2
            """
        )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                lines=7,
                placeholder="Type or paste a comment here...",
                label="Comment to analyze",
            )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Analyze", variant="primary")
            gr.Examples(
                examples=[
                    "Thanks for the helpful edit, this looks much better now.",
                    "You are a childish, pompous, self-regarding fool.",
                    "I will find you and hurt you if you do not stop.",
                ],
                inputs=text_input,
                label="Try an example",
            )
        with gr.Column(scale=1):
            output_html = gr.HTML(EMPTY_RESULTS_HTML, label="Results")

    submit_btn.click(fn=gradio_predict, inputs=text_input, outputs=output_html)
    text_input.submit(fn=gradio_predict, inputs=text_input, outputs=output_html)
    clear_btn.click(fn=lambda: ("", EMPTY_RESULTS_HTML), outputs=[text_input, output_html])

    with gr.Accordion("About this model & API usage", open=False):
        gr.Markdown(
            """
            **Model**: `distilbert-base-uncased`, fine-tuned on the
            [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
            dataset (~160K Wikipedia talk-page comments).

            **Performance** (macro-averaged, on the official held-out Kaggle test set, never used in training or tuning):

            | Metric | Score |
            |---|---|
            | ROC-AUC | 0.977 |
            | F1 | 0.546 |
            | Precision | 0.438 |

            **REST API**: `POST /analyze` with `{"text": "..."}`, or `POST /batch` with `{"texts": [...]}`.
            Full interactive API docs at [`/docs`](/docs).

            **Source, training notebooks, and full results**:
            [GitHub repository](https://github.com/Evanj0hn/toxicity-detection-distilbert)
            """
        )

app = gr.mount_gradio_app(app, demo, path="/", theme=gr.themes.Soft(), css=CUSTOM_CSS)
