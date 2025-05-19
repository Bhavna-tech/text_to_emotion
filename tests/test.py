from pathlib import Path
from model import EmotionDetector

detector = EmotionDetector.load(Path("artifacts"))
sentence = "I wasn't expecting that at all!"
label, conf = detector.predict(sentence)
print(f"{sentence}\n→ {label} ({conf:.3f})")