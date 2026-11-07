# Smart Fake News Detection System (Major Project)

## Project overview
A Natural Language Processing (NLP) and machine-learning application that converts news text into TF-IDF features, trains Logistic Regression and Multinomial Naive Bayes classifiers, compares their held-out performance, saves the selected model and fitted vectorizer using Joblib, and serves predictions through Streamlit. This follows Part B of the supplied InternsElite guideline (pages 6-12).

## Dataset: important distinction
The bundled `data/fake_news_dataset.csv` is a **synthetic demonstration dataset** created by `generate_demo_dataset.py`. Its 1,000 example texts were generated programmatically to test the end-to-end pipeline offline. They are not real news articles, and the generated labels do not establish the truth or falsity of real-world claims. Scores computed from this small synthetic corpus are only a software-execution demonstration. **Do not submit those scores as evidence of real-world fake-news detection performance.**

### Recommended real dataset option: WELFake
The WELFake record on Zenodo describes 72,134 articles (35,028 real and 37,106 fake), with `Title`, `Text`, and `Label` columns; labels are 0=fake and 1=real. Source: https://zenodo.org/records/4561253 (Verma, Agrawal & Prodan, 2021). The CSV is approximately 245 MB. Review the record, citation, and usage terms before redistribution or publication.

Download with internet access:

```powershell
python -m src.download_dataset
python -m src.train_model --data data/WELFake_Dataset.csv
```

The loader combines `Title` and `Text`, maps 0 to FAKE and 1 to REAL, and removes empty/duplicate records. It also accepts a CSV with `text,label` columns. If using another dataset, confirm its label meanings before training.

## Technologies
Python, NumPy, Pandas, Matplotlib, Scikit-learn, NLP, TF-IDF, Logistic Regression, Multinomial Naive Bayes, Joblib and Streamlit.

## Folder structure
```text
Smart_Fake_News_Detection/
├── data/fake_news_dataset.csv
├── notebooks/fake_news_analysis.ipynb
├── src/preprocessing.py
├── src/train_model.py
├── src/evaluate_model.py
├── src/download_dataset.py
├── src/predict_news.py
├── app/streamlit_app.py
├── models/model.joblib
├── models/tfidf_vectorizer.joblib
├── outputs/visualizations/
├── outputs/evaluation_results/
├── screenshots/
├── tests/test_pipeline.py
├── generate_demo_dataset.py
├── Project_Report.pdf
├── README.md
└── requirements.txt
```

## Setup and execution (Windows PowerShell / VS Code)
Open VS Code terminal in the `Smart_Fake_News_Detection` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train_model
python -m src.evaluate_model
python -m src.predict_news The city council published its monthly report today
streamlit run app/streamlit_app.py
```

If PowerShell blocks activation, run this once for the current user and open a new terminal:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

The first training command uses the bundled synthetic demonstration CSV. For a meaningful project evaluation, download WELFake and train with its path as shown above. Training is performed once; the Streamlit app loads the saved model and vectorizer and does not retrain when users submit text.

## Pipeline and leakage prevention
1. Load, inspect, clean and deduplicate labelled text.
2. Clean text with one shared function for training and user input.
3. Stratified 80/20 train-test split.
4. Fit TF-IDF on training text only; transform test text using the fitted vectorizer.
5. Train Logistic Regression and Multinomial Naive Bayes.
6. Compare accuracy, macro precision, macro recall, macro F1 and confusion matrices.
7. Select the higher macro-F1 model (accuracy breaks ties) for the demo app.
8. Save the selected classifier and fitted TF-IDF vectorizer with Joblib.

## Generated outputs
- `outputs/visualizations/label_distribution.png`
- `outputs/visualizations/word_count_distribution.png`
- `outputs/visualizations/confusion_matrix_*.png`
- `outputs/visualizations/model_comparison.png`
- `outputs/evaluation_results/evaluation_results.json`
- `outputs/evaluation_results/model_comparison.csv`
- `outputs/evaluation_results/test_predictions.csv`

## Tests
```powershell
python -m unittest discover -s tests -v
```

## Responsible interpretation and limitations
This is a text-classification tool, not a definitive fact-checker. It learns statistical patterns in the chosen labelled corpus and does not independently verify claims, publishers, dates, sources, images or live events. Dataset age, source selection, political/topic imbalance, language coverage, duplicate stories and label quality can bias results. Predictions may be incorrect, especially for satire, breaking news, unfamiliar subjects or text unlike the training data. Confirm important claims using reliable primary sources.

## Submission reminder
The supplied guideline sets the deadline as **10 November 2026** and requires each project in its own public GitHub repository. Include the source, dataset or dataset acquisition instructions, report PDF, README, dependencies and genuine output evidence. The Major Project also requires a short demonstration video and, if deployed, a live Streamlit link. Follow the email format announced in the official program group when submitting both repositories to support@internselite.com.
