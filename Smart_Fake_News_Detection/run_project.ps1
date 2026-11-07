$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.venv')) { python -m venv .venv }
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt
& $python -m src.train_model
& $python -m src.evaluate_model
& $python -m src.predict_news The city council published its monthly public transport report on Monday
& $python -m unittest discover -s tests -v
& $python -m streamlit run app/streamlit_app.py
