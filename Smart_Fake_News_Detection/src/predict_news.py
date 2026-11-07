"""Run one prediction from the command line using saved Joblib artifacts."""
from pathlib import Path
import argparse
import joblib
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.joblib"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+", help="Headline or article text to classify")
    args = parser.parse_args()
    text = clean_text(" ".join(args.text))
    if len(text.split()) < 3:
        raise SystemExit("Please provide at least three words of news text.")
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise SystemExit("Model files missing. Run `python -m src.train_model` first.")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    print(f"Model classification: LIKELY {prediction}")
    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(features)[0])
        print(f"Model confidence (not probability of truth): {confidence:.1%}")
    print("The system does not independently verify the claim or its source.")

if __name__ == "__main__": main()
