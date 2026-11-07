import unittest
from pathlib import Path
from src.preprocessing import clean_text
from src.train_model import load_dataset, DEFAULT_DATA

class FakeNewsTests(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("<b>Breaking</b> NEWS! https://example.com"), "breaking news")
        self.assertIn("not", clean_text("This is not verified"))
    def test_dataset_schema(self):
        data = load_dataset(DEFAULT_DATA)
        self.assertEqual(set(data.label.unique()), {"FAKE","REAL"})
        self.assertGreaterEqual(len(data), 100)

if __name__ == "__main__": unittest.main()
