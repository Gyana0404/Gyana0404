"""Streamlit front end for the saved Fake News classification model.
Run from project root: streamlit run app/streamlit_app.py
"""
from pathlib import Path
import sys, json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.preprocessing import clean_text
MODEL_PATH = ROOT/"models"/"model.joblib"
VECTORIZER_PATH = ROOT/"models"/"tfidf_vectorizer.joblib"
RESULTS_PATH = ROOT/"outputs"/"evaluation_results"/"evaluation_results.json"

st.set_page_config(page_title="Smart Fake News Detection", page_icon="📰", layout="wide")
st.title("Smart Fake News Detection System")
st.caption("InternsElite Major Project")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model files not found. Run `python -m src.train_model` from the project root first.")
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

try:
    model, vectorizer = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

metrics = {}
if RESULTS_PATH.exists():
    metrics = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Analyze a news headline or article")
    article = st.text_area("Paste news text", height=240, placeholder="Enter a headline or article text...")
    if st.button("Analyze news", type="primary", use_container_width=True):
        cleaned = clean_text(article)
        if not cleaned:
            st.warning("Please enter non-empty news text before submitting.")
        elif len(cleaned.split()) < 3:
            st.warning("Please provide a little more context (at least 3 words).")
        else:
            try:
                features = vectorizer.transform([cleaned])
                prediction = model.predict(features)[0]
                st.session_state["last_prediction"] = prediction
                st.session_state["last_confidence"] = float(max(model.predict_proba(features)[0])) if hasattr(model,"predict_proba") else None
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
    if "last_prediction" in st.session_state:
        pred = st.session_state["last_prediction"]
        conf = st.session_state.get("last_confidence")
        if pred == "FAKE": st.error("Model classification: LIKELY FAKE")
        else: st.success("Model classification: LIKELY REAL")
        if conf is not None: st.metric("Model confidence (not probability of truth)", f"{conf:.1%}")
with right:
    st.subheader("Model evaluation")
    selected = metrics.get("selected_model", "Not available")
    st.write(f"Selected model: **{selected}**")
    model_rows=[]
    for name, values in metrics.get("models",{}).items():
        model_rows.append({"Model":name,"Accuracy":values.get("accuracy"),"Precision":values.get("precision_macro"),"Recall":values.get("recall_macro"),"F1":values.get("f1_macro")})
    if model_rows:
        st.dataframe(pd.DataFrame(model_rows).set_index("Model").style.format("{:.3f}"), use_container_width=True)
    else:
        st.info("Run training to display model metrics.")

st.divider()
st.subheader("Confusion matrix and dataset overview")
model_key = selected
if model_key in metrics.get("models",{}):
    cm = metrics["models"][model_key]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["FAKE","REAL"], yticklabels=["FAKE","REAL"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Selected model confusion matrix")
    st.pyplot(fig); plt.close(fig)
counts = metrics.get("class_distribution",{})
if counts:
    st.write("Training dataset label counts")
    st.bar_chart(pd.Series(counts, name="records"))

st.warning("This model learns text patterns from its training dataset. It does not verify sources, check live events, or establish whether a claim is factually true. It may be wrong on satire, new events, unfamiliar topics, and text unlike its training data. Always verify important claims with reliable primary sources.")
