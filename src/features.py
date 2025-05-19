# from sklearn.feature_extraction.text import TfidfVectorizer

# def build_vectoriser() -> TfidfVectorizer:
#     return TfidfVectorizer(
#         strip_accents="unicode",
#         lowercase=True,
#         min_df=2,
#         sublinear_tf=True,
#         stop_words=None,
#     )
# #  return TfidfVectorizer()
from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectoriser():
    return TfidfVectorizer()
