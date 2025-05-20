from pathlib import Path
import joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from src.data_utils import normalise
from src.features import build_vectoriser

class EmotionDetector:
    """TF-IDF + LogisticRegression emotion classifier."""

    def __init__(self):
        self.vectoriser      = build_vectoriser()
        self.label_encoder   = LabelEncoder()
        self.model           = LogisticRegression(max_iter=1000)

    @staticmethod
    def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
        if "Clean" not in df:
            df = df.copy()
            df["Clean"] = df["Text"].apply(normalise)
        return df

    def fit(self, df: pd.DataFrame):
        df = self._prep_df(df)
        X = self.vectoriser.fit_transform(df["Clean"])
        y = self.label_encoder.fit_transform(df["Emotion"])
        self.model.fit(X, y)

    def evaluate(self, df: pd.DataFrame):
        df = self._prep_df(df)
        X = self.vectoriser.transform(df["Clean"])
        y = self.label_encoder.transform(df["Emotion"])
        y_pred = self.model.predict(X)
        print("\nAccuracy:", round(accuracy_score(y, y_pred), 3))
        print("\nClassification Report:\n",
              classification_report(y, y_pred,
                                    target_names=self.label_encoder.classes_))

    def predict(self, text: str):
        clean = normalise(text)
        vec   = self.vectoriser.transform([clean])
        probs = self.model.predict_proba(vec)[0]
        idx   = probs.argmax()
        return self.label_encoder.inverse_transform([idx])[0], round(probs[idx], 3)
            
    def save(self, folder="src/artifacts"):
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model,        p / "model.joblib")
        joblib.dump(self.vectoriser,   p / "vectoriser.joblib")
        joblib.dump(self.label_encoder,p / "label_encoder.joblib")

    @classmethod
    def load(cls, folder="src/artifacts"):
        p        = Path(folder)
        self     = cls.__new__(cls)
        self.model         = joblib.load(p / "model.joblib")
        self.vectoriser    = joblib.load(p / "vectoriser.joblib")
        self.label_encoder = joblib.load(p / "label_encoder.joblib")
        return self
