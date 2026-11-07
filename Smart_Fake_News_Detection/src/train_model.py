"""Train and compare Logistic Regression and Multinomial Naive Bayes.

Default: bundled synthetic demo data.
Real-data option: python -m src.download_dataset, then
python -m src.train_model --data data/WELFake_Dataset.csv
"""
from pathlib import Path
import argparse, json, re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "fake_news_dataset.csv"
MODELS = ROOT / "models"
VIZ = ROOT / "outputs" / "visualizations"
EVAL = ROOT / "outputs" / "evaluation_results"


def normalize_label(value):
    s = str(value).strip().lower()
    if s in {"fake", "false", "0", "0.0", "f"}: return "FAKE"
    if s in {"real", "true", "1", "1.0", "r"}: return "REAL"
    raise ValueError(f"Unsupported label {value!r}; expected FAKE/REAL or 0/1 (0=FAKE, 1=REAL).")


def load_dataset(path):
    df = pd.read_csv(path)
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    if "label" not in lower_map:
        raise ValueError("Dataset needs a label/Label column.")
    label_col = lower_map["label"]
    # Supports WELFake columns Title, Text, Label and compact text,label files.
    text_parts = []
    for key in ("title", "text", "content", "article"):
        if key in lower_map:
            text_parts.append(df[lower_map[key]].fillna("").astype(str))
    if not text_parts:
        raise ValueError("Dataset needs at least one text column: text, Text, title, or content.")
    combined = text_parts[0]
    for part in text_parts[1:]:
        combined = combined + " " + part
    data = pd.DataFrame({"text": combined, "label": df[label_col]})
    data = data[data["label"].notna()].copy()
    data["text"] = data["text"].map(clean_text)
    data["label"] = data["label"].map(normalize_label)
    data = data[(data.text.str.len() > 0) & data.label.notna()].copy()
    data = data.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if data.label.nunique() != 2:
        raise ValueError("After cleaning, the dataset must contain both FAKE and REAL labels.")
    return data


def plot_eda(data):
    VIZ.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    counts = data.label.value_counts().reindex(["FAKE","REAL"])
    plt.figure(figsize=(7,5)); counts.plot(kind="bar")
    plt.title("Fake and Real News Label Distribution"); plt.xlabel("Class"); plt.ylabel("Article count")
    plt.tight_layout(); plt.savefig(VIZ/"label_distribution.png", dpi=150); plt.close()
    data = data.copy(); data["word_count"] = data.text.str.split().str.len()
    plt.figure(figsize=(8,5))
    for label in ["FAKE", "REAL"]:
        plt.hist(data.loc[data.label == label, "word_count"], bins=25, alpha=0.55, label=label)
    plt.legend(); plt.title("Article Word Count by Label"); plt.xlabel("Word count")
    plt.tight_layout(); plt.savefig(VIZ/"word_count_distribution.png", dpi=150); plt.close()
    summary = data.groupby("label").word_count.agg(["count","mean","median","std"]).reset_index()
    summary.to_csv(EVAL/"text_summary_by_label.csv", index=False)


def metric_pack(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true,y_pred)),
        "precision_macro": float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
        "recall_macro": float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
        "f1_macro": float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
        "confusion_matrix_labels_order": ["FAKE", "REAL"],
        "confusion_matrix": confusion_matrix(y_true,y_pred,labels=["FAKE","REAL"]).tolist(),
    }


def main(data_path=DEFAULT_DATA):
    data = load_dataset(data_path)
    plot_eda(data)
    X_train, X_test, y_train, y_test = train_test_split(
        data.text, data.label, test_size=0.20, random_state=42, stratify=data.label
    )
    # Fit vocabulary on training data only; transform held-out data with same fitted vectorizer.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=30000, min_df=1)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
    }
    metrics = {}
    fitted = {}
    predictions = {}
    for name, clf in algorithms.items():
        clf.fit(X_train_tfidf, y_train)
        pred = clf.predict(X_test_tfidf)
        fitted[name] = clf
        predictions[name] = pred
        metrics[name] = metric_pack(y_test, pred)
        metrics[name]["classification_report"] = classification_report(y_test,pred,labels=["FAKE","REAL"],output_dict=True,zero_division=0)
        cm = np.array(metrics[name]["confusion_matrix"])
        plt.figure(figsize=(5.5,4.5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["FAKE","REAL"], yticklabels=["FAKE","REAL"])
        plt.xlabel("Predicted label"); plt.ylabel("Actual label"); plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout(); plt.savefig(VIZ/("confusion_matrix_"+name.lower().replace(" ","_") + ".png"), dpi=150); plt.close()
    # Choose higher macro F1 on the same held-out comparison set, as requested in the brief.
    selected_name = max(metrics, key=lambda name: (metrics[name]["f1_macro"], metrics[name]["accuracy"]))
    selected_model = fitted[selected_name]
    MODELS.mkdir(parents=True, exist_ok=True); EVAL.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODELS/"model.joblib")
    joblib.dump(vectorizer, MODELS/"tfidf_vectorizer.joblib")
    comparison = pd.DataFrame([
        {"model": name, **{k: metrics[name][k] for k in ["accuracy","precision_macro","recall_macro","f1_macro"]}}
        for name in metrics
    ])
    comparison.to_csv(EVAL/"model_comparison.csv", index=False)
    plt.figure(figsize=(8,5)); comparison.set_index("model")[["accuracy","precision_macro","recall_macro","f1_macro"]].plot(kind="bar", ax=plt.gca())
    plt.ylim(0,1.05); plt.ylabel("Score"); plt.title("Classification Model Comparison"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(VIZ/"model_comparison.png", dpi=150); plt.close()
    pred_df = pd.DataFrame({"text": X_test.to_numpy(), "actual_label": y_test.to_numpy()})
    for name, pred in predictions.items(): pred_df["predicted_"+name.lower().replace(" ","_")] = pred
    pred_df.to_csv(EVAL/"test_predictions.csv", index=False)
    report = {
        "dataset_path": str(Path(data_path).name),
        "dataset_note": "Synthetic demonstration dataset" if Path(data_path).name == "fake_news_dataset.csv" else "External dataset supplied by user/downloaded from source",
        "total_clean_records": int(len(data)),
        "class_distribution": {k:int(v) for k,v in data.label.value_counts().items()},
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "tfidf_features": int(X_train_tfidf.shape[1]),
        "models": metrics,
        "selected_model": selected_name,
        "selection_rule": "Highest macro F1 on the held-out comparison set; accuracy breaks ties.",
    }
    with open(EVAL/"evaluation_results.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print("SMART FAKE NEWS PROJECT TRAINING COMPLETE")
    print(json.dumps({k:v for k,v in report.items() if k != "models"}, indent=2))
    for name in metrics:
        print(f"\n{name}: {json.dumps({k:metrics[name][k] for k in ['accuracy','precision_macro','recall_macro','f1_macro']})}")
    print(f"Selected model: {selected_name}")
    print(f"Saved: {MODELS/'model.joblib'} and {MODELS/'tfidf_vectorizer.joblib'}")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV path; supports text,label or Title,Text,Label schema")
    args = parser.parse_args()
    main(Path(args.data))
