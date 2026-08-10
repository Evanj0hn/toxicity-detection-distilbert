# Hate Speech and Toxicity Detection Using DistilBERT

A multi-label toxicity classification system, built for PROG74040 (Advanced Topics in Artificial Intelligence and Machine Learning), Conestoga College, Spring 2026, Section 1.

**Team (Group 2):** Evan John Tomy (8884866), Jerin Pious (add student ID)

## Problem

Classifies a comment into six toxicity categories at once: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. Built on the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset (~160K Wikipedia talk-page comments).

## Results (on the official, held-out Kaggle test set, never used in training or tuning)

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.965 | 0.423 | 0.297 | 0.840 |
| LSTM + GloVe | 0.951 | 0.211 | 0.128 | 0.944 |
| **DistilBERT (fine-tuned, tuned thresholds)** | **0.977** | **0.546** | **0.438** | 0.789 |

Full breakdown, per-label metrics, and confusion matrices in `05_evaluation_comparison.ipynb`.

## Notebooks (run in order)

1. `01_data_exploration.ipynb`: verifies dataset structure, class distribution, and content quality
2. `02_preprocessing.ipynb`: cleaning, tokenization, stratified train/val split, class weights, and a documented data-augmentation experiment (tested, found to hurt results, reverted)
3. `03_baseline_models.ipynb`: TF-IDF + Logistic Regression, LSTM + GloVe
4. `04_distilbert_finetuning.ipynb`: fine-tunes `distilbert-base-uncased`, tunes decision thresholds, generates Captum explainability
5. `05_evaluation_comparison.ipynb`: final head-to-head comparison on the untouched test set

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch transformers captum sentencepiece iterative-stratification jupyter
```

Requires an NVIDIA GPU with CUDA for practical training time (DistilBERT fine-tuning and LSTM training use `torch.cuda`).

## Data and model files not included in this repository

To keep this repository lightweight (GitHub blocks any file over 100MB, and the DistilBERT weights alone are 256MB), the following are excluded via `.gitignore` and must be regenerated locally:

| File / folder | How to get it |
|---|---|
| `train.csv/`, `test.csv/`, `test_labels.csv/` | Download from the [Kaggle competition page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) |
| `glove.6B.zip`, `glove/` | Download from the [Stanford NLP GloVe project](https://nlp.stanford.edu/projects/glove/) |
| `train_clean.csv`, `val_clean.csv`, `pos_weight.pt` | Regenerate by running `02_preprocessing.ipynb` |
| `model_artifact/`, `lstm_model.pt` | Regenerate by running `03_baseline_models.ipynb` and `04_distilbert_finetuning.ipynb` |

## Deployment

**Live demo and API**: [huggingface.co/spaces/EvanJ0hn/toxicity-detection-distilbert](https://huggingface.co/spaces/EvanJ0hn/toxicity-detection-distilbert) (interactive Gradio UI at the root; REST API at `/analyze` and `/batch`, docs at `/docs`).

Deployed as a Docker-SDK Hugging Face Space (FastAPI + Gradio in one container), rather than the AWS EC2 setup originally planned in `Group-2-Phase-1-Project-Proposal-Jerin-Evan.pdf`. This is a documented deviation: EC2 access here was only available through AWS Academy's session-based Learner Lab, which stops resources between sessions rather than staying continuously live; Hugging Face Spaces is explicitly listed as an acceptable deployment target in the course's Evaluation Table and gives a persistent public URL with less infrastructure overhead. Source for the deployed app is in `hf_space/`.

## Team Contributions

| Member | Responsibilities |
|---|---|
| Jerin Pious | Model architecture, fine-tuning, evaluation, explainability |
| Evan Tomy | Data preprocessing pipeline, deployment infrastructure |
