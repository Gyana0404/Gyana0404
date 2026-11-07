"""Text cleaning shared by training and inference."""
import html
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
_SPACE_RE = re.compile(r"\s+")

def clean_text(value):
    """Normalize text consistently without removing negation words."""
    if value is None:
        return ""
    text = html.unescape(str(value)).lower()
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()
