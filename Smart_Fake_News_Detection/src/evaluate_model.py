"""Print the evaluation JSON produced by src.train_model."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
path = ROOT/"outputs"/"evaluation_results"/"evaluation_results.json"
if not path.exists():
    raise SystemExit("Evaluation not found. Run: python -m src.train_model")
report = json.loads(path.read_text(encoding="utf-8"))
for model, result in report["models"].items():
    print(model)
    for metric in ("accuracy","precision_macro","recall_macro","f1_macro"):
        print(f"  {metric}: {result[metric]:.4f}")
print("Selected:", report["selected_model"])
