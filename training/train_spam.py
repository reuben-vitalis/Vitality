"""Train the Vitality Spam & Phishing model.

Run from the project root:
    python -m training.train_spam

Reads  data/spam.csv  (columns: text, label = "spam" | "safe")
Writes models/spam_model.joblib
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spam.csv"
MODEL = ROOT / "models" / "spam_model.joblib"


def build_pipeline() -> Pipeline:
    """Word features (1-2 word phrases) + character features (3-5 letter chunks) -> Logistic Regression."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(max_iter=3000, C=5))])


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df["text"].astype(str), (df["label"] == "spam").astype(int)
    print(f"Loaded {len(df)} messages: {y.sum()} spam/phishing, {len(y) - y.sum()} safe\n")

    # 1) Cross-validation: train and test 5 times on different splits, then average.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(build_pipeline(), X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1"])
    print("5-fold cross-validation (average):")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {metric:<9} {scores['test_' + metric].mean():.3f}")

    # 2) One held-out test set (20%) for a readable report and confusion matrix.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = build_pipeline().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print("\nHeld-out test set (20%):")
    print(classification_report(y_te, pred, target_names=["safe", "spam"], digits=3))
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"Confusion matrix: {tp} scams caught, {fn} scams missed, {tn} safe correct, {fp} false alarms\n")

    # 3) Final model trained on ALL data, saved for the app.
    final = build_pipeline().fit(X, y)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": final,
                 "cv_accuracy": round(float(scores["test_accuracy"].mean()), 3),
                 "n_messages": int(len(df))}, MODEL)
    print(f"Saved model to {MODEL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
