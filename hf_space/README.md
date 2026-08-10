---
title: Toxicity Detection DistilBERT
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Toxicity Detection (DistilBERT)

Multi-label toxicity classifier fine-tuned on the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset. Built for PROG74040 (Advanced Topics in Artificial Intelligence and Machine Learning), Conestoga College, Spring 2026.

**Team (Group 2):** Evan John Tomy, Jerin Pious

Detects six categories at once: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

## Using this Space

- **Interactive demo**: type a comment into the text box above and see per-category scores.
- **REST API**: `POST /analyze` with `{"text": "..."}`, or `POST /batch` with `{"texts": [...]}`. Full API docs at `/docs`.

Full project, training notebooks, and evaluation results: see the main [GitHub repository](https://github.com/Evanj0hn/toxicity-detection-distilbert).
