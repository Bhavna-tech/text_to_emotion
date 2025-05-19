# from pathlib import Path
# import joblib
# import numpy as np
# from scipy.special import softmax
# from sklearn.svm import LinearSVC
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# from data_utils import normalise
# from features import build_vectoriser


# class EmotionDetector:
#     def __init__(self):
#         self.tfidf = build_vectoriser()
#         self.encoder = LabelEncoder()
#         base_svc = LinearSVC(C=0.5, max_iter=10_000)
#         self.clf = CalibratedClassifierCV(base_svc, cv=5)

#    # Training model 
#     def fit(self, texts, labels) -> None:
#         X = self.tfidf.fit_transform(texts)
#         y = self.encoder.fit_transform(labels)
#         self.clf.fit(X, y)

#     # Evaluation of model
#     def evaluate(self, texts, labels) -> str:
#         X = self.tfidf.transform(texts)
#         y_true = self.encoder.transform(labels)
#         y_pred = self.clf.predict(X)
#         acc = accuracy_score(y_true, y_pred)
#         rep = classification_report(
#             y_true, y_pred, target_names=self.encoder.classes_
#         )
#         return f"Accuracy: {acc:.3f}\n\n{rep}"

#     # Prediction
#     def predict(self, sentence: str):
#         clean = normalise(sentence)
#         X = self.tfidf.transform([clean])
#         probas = self.clf.predict_proba(X)[0]
#         idx = int(np.argmax(probas))
#         return self.encoder.inverse_transform([idx])[0], probas[idx]

#     # saving the model using joblib
#     def save(self, folder: Path) -> None:
#         folder.mkdir(parents=True, exist_ok=True)
#         joblib.dump(self.tfidf, folder / "tfidf.joblib")
#         joblib.dump(self.encoder, folder / "label_encoder.joblib")
#         joblib.dump(self.clf, folder / "calibrated_svc.joblib")

#     @classmethod
#     def load(cls, folder: Path):
#         self = cls.__new__(cls)
#         self.tfidf = joblib.load(folder / "tfidf.joblib")
#         self.encoder = joblib.load(folder / "label_encoder.joblib")
#         self.clf = joblib.load(folder / "calibrated_svc.joblib")
#         return self

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from .features import build_vectoriser
from .data_utils import normalise
import joblib
from pathlib import Path

class EmotionDetector:
    def __init__(self):
        self.vectoriser = build_vectoriser()
        self.label_encoder = LabelEncoder()
        self.model = CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=10000))

    def preprocess(self, df):
        df["Clean"] = df["Text"].apply(normalise)
        return df

    def fit(self, df):
        df = self.preprocess(df)
        X = self.vectoriser.fit_transform(df["Clean"])
        y = self.label_encoder.fit_transform(df["Emotion"])
        self.model.fit(X, y)

    def evaluate(self, df):
        df = self.preprocess(df)
        X = self.vectoriser.transform(df["Clean"])
        y = self.label_encoder.transform(df["Emotion"])
        y_pred = self.model.predict(X)
        print("\nAccuracy:", accuracy_score(y, y_pred))
        print("\nClassification Report:\n", classification_report(y, y_pred, target_names=self.label_encoder.classes_))

    def predict(self, text):
        clean = normalise(text)
        vec = self.vectoriser.transform([clean])
        probs = self.model.predict_proba(vec)[0]
        pred_index = probs.argmax()
        label = self.label_encoder.inverse_transform([pred_index])[0]
        confidence = round(probs[pred_index], 3)
        return label, confidence

    def save(self, path_prefix="artifacts"):
        folder = Path(path_prefix)
        folder.mkdir(parents=True, exist_ok=True) 
        joblib.dump(self.model, folder/"model.joblib")
        joblib.dump(self.vectoriser, folder/"vectoriser.joblib")
        joblib.dump(self.label_encoder, folder/"label_encoder.joblib")
