import unittest
import joblib
from pathlib import Path
from src.preprocessing import clean_text
from src.train_model import ROOT

class SavedArtifactTests(unittest.TestCase):
    def test_saved_model_predicts(self):
        model_path = ROOT / 'models' / 'model.joblib'
        vectorizer_path = ROOT / 'models' / 'tfidf_vectorizer.joblib'
        self.assertTrue(model_path.exists())
        self.assertTrue(vectorizer_path.exists())
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        sample = clean_text('The city council published its monthly public transport report on Monday')
        pred = model.predict(vectorizer.transform([sample]))[0]
        self.assertIn(pred, {'FAKE', 'REAL'})

if __name__ == '__main__': unittest.main()
