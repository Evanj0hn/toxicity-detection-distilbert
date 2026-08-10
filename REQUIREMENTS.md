# Requirements & Grading Checklist — Hate Speech and Toxicity Detection Using DistilBERT

**Group 2 — Jerin Pious, Evan Tomy — PROG74040, Conestoga College**

Related: [PROJECT_PLAN.md](PROJECT_PLAN.md) — roadmap, architecture, team workflow.

## 1. Purpose of This Document

This is the specification and acceptance-criteria checklist for Phase II. Every requirement below traces back to `Group-2-Phase-1-Project-Proposal-Jerin-Evan.pdf`, `Course Project Guidelines Phase II.pdf`, `Phase_II_Project_Rubric.pdf`, `Evaluation Table.pdf`, or `What to hand in.txt`. Check items off as they're actually verified working — not as they're merely attempted. Section 9 is the part that most directly protects your grade; review it before submission regardless of how the rest of the build goes.

## 2. Functional Requirements

| ID | Requirement | Source | Status |
|---|---|---|---|
| FR-1 | Accept raw text input (comment/post/sentence) via a REST API endpoint | Proposal §3.1 | ⬜ |
| FR-2 | Classify input across all six toxicity categories in a single forward pass | Proposal §3.1 | ⬜ |
| FR-3 | Return per-class confidence scores (float, 0.0–1.0) as a JSON payload | Proposal §3.1, §4.3 | ⬜ |
| FR-4 | `POST /analyze` accepts `{"text": "..."}`, returns `{"scores": {...6 keys...}, "flagged": {...}}` | Proposal §4.3 | ⬜ |
| FR-5 | `POST /batch` accepts an array of strings, returns the same structure per item | Proposal §4.3 | ⬜ |
| FR-6 | `flagged` applies a per-category threshold to `scores` to produce a binary decision | Proposal §4.3 | ⬜ |
| FR-7 | React frontend: text submission form + per-category result display | Proposal §4.2 | ⬜ |

## 3. Non-Functional Requirements

| ID | Requirement | Target | Source | Status |
|---|---|---|---|---|
| NFR-1 | Inference latency | < 500 ms per warm request on CPU | Proposal Table 3 | ⬜ |
| NFR-2 | Model memory footprint on disk | < 300 MB (DistilBERT) | Proposal Table 3 | ⬜ |
| NFR-3 | Availability | Single EC2 instance, best-effort uptime | Proposal Table 3 | ⬜ |
| NFR-4 | Cost | AWS Free Tier only — zero ongoing spend | Proposal Table 3 | ⬜ |
| NFR-5 | Environment reproducibility | Docker container, no dependency drift | Proposal Table 6 | ⬜ |
| NFR-6 | CI | GitHub Actions runs on every push | Proposal Table 6 | ⬜ |
| NFR-7 | Out-of-memory safety on 1GB RAM instance | 4GB Linux swap configured on EBS | Proposal §4.1 | ⬜ |

## 4. Data Requirements

| ID | Requirement | Status |
|---|---|---|
| D-1 | Dataset acquired: Jigsaw Toxic Comment Classification Challenge (Kaggle, CC0) | ✅ `train.csv/`, `test.csv/`, `test_labels.csv/` present locally, schema verified |
| D-2 | Rows in `test_labels.csv` with `-1` placeholders excluded from any test-set scoring | ⬜ |
| D-3 | Text cleaning: strip HTML tags, normalize Unicode | ⬜ |
| D-4 | Tokenization via Hugging Face DistilBERT tokenizer | ⬜ |
| D-5 | Sequence truncation/padding to 256 tokens | ⬜ |
| D-6 | Labels encoded as multi-hot binary arrays (length 6) | ⬜ |
| D-7 | Class imbalance addressed via class-weighted loss + stratified train/val split | ⬜ |
| D-8 | Data augmentation for minority classes: back-translation (EN→FR→EN) and/or LLM-paraphrased synthetic samples | ⬜ — flagged as *planned*, not committed, in Phase 1; confirm whether it shipped |
| D-9 | Final train/val/test split ratios documented in the report | ⬜ |

## 5. Model Requirements

### 5.1 Baseline models (at least one required by guidelines; proposal commits to two)

| ID | Requirement | Status |
|---|---|---|
| M-1 | TF-IDF + Logistic Regression baseline, trained and scored | ⬜ |
| M-2 | LSTM + GloVe (100-dim embeddings) baseline, trained and scored | ⬜ |
| M-3 | Both baselines' metrics recorded for later comparison (not just "it ran") | ⬜ |

### 5.2 Advanced model

| ID | Requirement | Status |
|---|---|---|
| M-4 | Base model: `distilbert-base-uncased` | ⬜ |
| M-5 | Multi-label sigmoid output head (not softmax) | ⬜ |
| M-6 | Loss: `BCEWithLogitsLoss` | ⬜ |
| M-7 | Optimizer: AdamW, learning rate 2e-5 | ⬜ |
| M-8 | Dropout 0.3, 3–5 epochs, batch size 16, max sequence length 256 | ⬜ |
| M-9 | Fine-tuned weights exported and versioned (not left only in Colab runtime) | ⬜ |
| M-10 | Sigmoid decision threshold swept 0.3–0.6 per class, optimized on validation F1 | ⬜ |

### 5.3 Explainability / advanced techniques (source of "Technical Depth and Innovation" points)

| ID | Requirement | Status |
|---|---|---|
| M-11 | SHAP or Integrated Gradients token-level attribution implemented | ⬜ |
| M-12 | Explainability output surfaced somewhere demo-able (frontend or report figure), not just a notebook cell | ⬜ |
| M-13 | *(Optional, exploratory)* Multi-task learning variant (shared encoder, six task-specific heads) evaluated against the single multi-label head | ⬜ |

## 6. Evaluation Requirements

| ID | Requirement | Target | Status |
|---|---|---|---|
| E-1 | ROC-AUC (macro) reported for every model | > 0.95 for DistilBERT | ⬜ |
| E-2 | F1-score (macro) reported for every model | > 0.82 for DistilBERT | ⬜ |
| E-3 | Precision/recall reported per class (all 6), including the rare `threat` class | — | ⬜ |
| E-4 | Confusion matrix per toxicity category | Visual | ⬜ |
| E-5 | **Baseline vs. advanced model comparison presented side by side** (explicit grading criterion) | Table/figure with commentary on which wins and why | ⬜ |
| E-6 | Results explained in prose, not just tables — does the improved model actually beat baseline, and by how much | — | ⬜ |

## 7. Deployment Requirements

| ID | Requirement | Status |
|---|---|---|
| DP-1 | Model deployed on an external cloud service (AWS/Azure/GCP) — required to hit top rubric band, not just "attempted" | ⬜ |
| DP-2 | Deployed endpoint is **externally reachable**, not just `localhost` | ⬜ |
| DP-3 | Docker container built and running on EC2 t2.micro | ⬜ |
| DP-4 | FastAPI backend serving `/analyze` and `/batch` end-to-end against the live model | ⬜ |
| DP-5 | React frontend deployed and able to call the live API | ⬜ |
| DP-6 | Deployment documented clearly enough that a grader can use it without help (README, or report section) | ⬜ |
| DP-7 | Known limitations of the deployed system explicitly discussed (cold start, latency, uptime, etc.) | ⬜ |

## 8. Documentation & Submission Requirements

### 8.1 File naming (per `What to hand in.txt`, adapted to this team)

| Deliverable | Filename |
|---|---|
| Report | `Group-2-Phase-2-Report-Jerin-Evan.pdf` |
| Code | `Group-2-Phase-2-code-Jerin-Evan.ipynb` |
| Presentation | `Group-2-Phase-2-Presentation-Jerin-Evan.pptx` |

`Course Project Guidelines Phase II.pdf` separately asks for code "in a GitHub repository or on Google Cloud" with a README and a link — this doesn't cleanly match "submit an `.ipynb`." **Submit both**: the notebook file under the name above, and a README + repo/cloud link inside the report, to satisfy either grading path.

### 8.2 Report content checklist (≤10 pages) — verbatim sections required by the guidelines

- [ ] Project Progress Summary (topic, NLP problem, progress since Phase I)
- [ ] Dataset Collection and Preprocessing
- [ ] Updated System Architecture (how it's *actually* being built, not the Phase 1 plan restated)
- [ ] Baseline Model (implementation description)
- [ ] Advanced Model (description of the fine-tuned model)
- [ ] Preliminary Evaluation Results (baseline **and** advanced, if available)
- [ ] Deployment (hosting platform + how the end user uses it)
- [ ] Team Workflow and Contribution (individual responsibilities and actual contributions)

### 8.3 Presentation content checklist (15-minute pitch)

- [ ] Project title and team members
- [ ] Problem recap
- [ ] Dataset summary
- [ ] Data preprocessing pipeline
- [ ] Baseline model
- [ ] Advanced model design
- [ ] Current implementation progress
- [ ] Preliminary results
- [ ] Updated system architecture
- [ ] Deployment progress
- [ ] Challenges and risks
- [ ] Next steps for the final phase

### 8.4 Code submission checklist

- [ ] Dataset loading code
- [ ] Preprocessing code
- [ ] Baseline model code
- [ ] Advanced model code
- [ ] Evaluation code
- [ ] README explaining how to run it
- [ ] Link to GitHub repo or Google Cloud project included in the report

## 9. Grading Alignment Matrix

This is the section to review right before submission. It quotes the **top band** language directly from the grading documents so there's no ambiguity about what "full marks" actually requires.

### 9.1 `Phase_II_Project_Rubric.pdf` (20% of course grade: 4% individual + 16% group)

| Criterion | Weight | Top-band requirement (verbatim) | Status |
|---|---|---|---|
| A. Individual Content Understanding and Presentation | 4% (individual) | "Explains individual contribution clearly, answers questions well, and demonstrates strong research effort and deep thinking." | ⬜ — applies to **each** team member separately |
| B1. Overall Project Progress and Requirements | 4% | "All requirements are fully addressed... strong progress, clear milestones, well-defined backlog items, and strong evidence that the team is on track for final delivery." | ⬜ |
| B2. Technical Implementation and Model Development | 4% | "A strong working prototype with clear dataset processing, baseline and advanced model implementation, meaningful evaluation results, and thoughtful technical decisions." | ⬜ |
| B3. Presentation Quality, Preparedness, and Team Coordination | 4% | "Highly organized, clear, engaging, visually appealing, well rehearsed... all team members demonstrate strong coordination and preparation." | ⬜ |
| B4. Model Deployment | 4% | "Deployed on an external cloud service... with clear documentation on how to interact with it. The model is fully accessible and functional." | ⬜ |

Note the trap in B4's grading band below top: *"Model is deployed on a cloud service, but minor issues are present, or documentation lacks some detail"* only scores 2/3. A working-but-undocumented deployment does not get full marks — the README/report explanation matters as much as the deployment itself.

### 9.2 `Evaluation Table.pdf` (16 points, presentation-day scoring)

| Criterion | Points | Full-marks requirement (verbatim/paraphrase) | Status |
|---|---|---|---|
| Participation | 1 | Attend and actively ask questions during other groups' presentations | ⬜ |
| Project Guidelines and Required Materials | 2 | All materials (slides, report, code/notebook, links) in correct format, well organized, on time | ⬜ |
| Problem Definition & Impact | 1 | Clearly articulated, impactful, appropriate for advanced ML, strong motivation | ⬜ |
| Implement of Baseline Model | 2 | Functional, clearly explained, used as a reference point | ⬜ |
| Implement of Improved or Finetuned Model | 3 | Correct, well explained, connected to project goal; training/fine-tuning details (dataset prep, model choice, hyperparameters, training process) described | ⬜ |
| Experimental Results: Comparison | 3 | Baseline vs. improved model compared with metrics/tables/figures; explains what results mean and whether improvement is real | ⬜ |
| Technical Depth and Innovation | 1 | Thoughtful model selection, error analysis, data augmentation, hyperparameter tuning, or similar — SHAP explainability satisfies this if actually shown | ⬜ |
| Deployment | 3 | Usable demo (web app/API/etc.) that lets users test the model; limitations discussed | ⬜ |
| **Total** | **16** | | |

### 9.3 Reconciling the two grading documents

Both appear to score the same Phase II presentation/submission from different angles — the rubric in weighted categories, the evaluation table in point totals. They overlap almost entirely on **baseline model, advanced model, comparison, deployment, and presentation quality**. Treat the evaluation table as the tactical checklist during the presentation and the rubric as the strategic view of where the 20% actually comes from. There is no item that appears in one and meaningfully contradicts the other.

## 10. Definition of Done — Phase II Submission Gate

Do not submit until every box below is checked:

- [ ] Baseline model runs end-to-end and its metrics are recorded
- [ ] DistilBERT fine-tuned model runs end-to-end and its metrics are recorded
- [ ] Baseline vs. DistilBERT comparison exists as a table/figure with written interpretation
- [ ] Deployed model is reachable from a URL that is not `localhost`, right now, by someone other than the person who built it
- [ ] Report is ≤10 pages and hits all 8 required sections (§8.2)
- [ ] Presentation hits all 12 required components (§8.3) and fits in 15 minutes when rehearsed
- [ ] Code submitted as both the named `.ipynb` and a repo/cloud link with README (§8.1)
- [ ] Both team members can individually explain the full pipeline, not just their own component (rubric criterion A is scored per person)
- [ ] All three files renamed to the exact convention in §8.1 before upload
- [ ] Phase II due date confirmed with the instructor (not assumed from the Phase 1 internal timeline)
