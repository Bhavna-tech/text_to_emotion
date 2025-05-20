from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectoriser():
    """Return a plain word-level TF-IDF vectoriser."""
    return TfidfVectorizer()
