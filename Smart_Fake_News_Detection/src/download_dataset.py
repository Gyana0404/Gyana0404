"""Download WELFake dataset from its Zenodo record (requires internet)."""
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/records/4561253/files/WELFake_Dataset.csv?download=1"
DEST = ROOT / "data" / "WELFake_Dataset.csv"

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading WELFake dataset (~245 MB). This may take time.")
    urlretrieve(URL, DEST)
    print(f"Saved dataset to: {DEST}")
    print("Run: python -m src.train_model --data data/WELFake_Dataset.csv")

if __name__ == "__main__": main()
