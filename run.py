from pathlib import Path
from src.data_utils import load_emotion_txt
from src.model import EmotionDetector
import pandas as pd

DATA = Path("data")
train_df = load_emotion_txt(DATA / "train.txt")
val_df = load_emotion_txt(DATA / "val.txt")
test_df = load_emotion_txt(DATA / "test.txt")

full_train_df = pd.concat([train_df, val_df], ignore_index=True)

clf = EmotionDetector()
clf.fit(full_train_df)
clf.evaluate(test_df)

text = "I'm so happy today!"
label, confidence = clf.predict(text)
print(f"\nInput: {text}\nPredicted Emotion: {label} ({confidence})")

clf.save("outputs")
