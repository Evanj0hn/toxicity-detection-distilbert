import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def compute_metrics(y_true, y_prob, y_pred, label_cols, model_name, verbose=True):
    rows = []
    for i, label in enumerate(label_cols):
        rows.append({
            "model": model_name,
            "label": label,
            "roc_auc": roc_auc_score(y_true[:, i], y_prob[:, i]),
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
        })
    df = pd.DataFrame(rows)
    if verbose:
        macro = df[["roc_auc", "f1", "precision", "recall"]].mean()
        print(f"=== {model_name} — per-label ===")
        print(df.set_index("label").round(4))
        print(f"\n=== {model_name} — macro-averaged ===")
        print(macro.round(4))
    return df
