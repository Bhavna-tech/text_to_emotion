# from pathlib import Path
# from src.data_utils import load_emotion_txt
# from src.model import EmotionDetector
# import pandas as pd

# DATA = Path("data")
# train_df = load_emotion_txt(DATA / "train.txt")
# val_df = load_emotion_txt(DATA / "val.txt")
# test_df = load_emotion_txt(DATA / "test.txt")

# full_train_df = pd.concat([train_df, val_df], ignore_index=True)

# clf = EmotionDetector()
# clf.fit(full_train_df)
# clf.evaluate(test_df)

# text = "I'm so happy today!"
# label, confidence = clf.predict(text)
# print(f"\nInput: {text}\nPredicted Emotion: {label} ({confidence})")

# clf.save("outputs")
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import pandas as pd
from src.data_utils import load_emotion_txt
from src.model import EmotionDetector
# import pandas as pd
# from pathlib import Path
# from src.data_utils import load_emotion_txt
# from src.model import EmotionDetector

DATA = Path("data")
train_df = load_emotion_txt(DATA / "train.txt")
val_df   = load_emotion_txt(DATA / "val.txt")
test_df  = load_emotion_txt(DATA / "test.txt")

# Train on train + val
full_train = pd.concat([train_df, val_df], ignore_index=True)

clf = EmotionDetector()
clf.fit(full_train)
print("\n--- Test set evaluation ---")
clf.evaluate(test_df)

# Demo
while True:
    txt = input("\nEnter text (q to quit): ")
    if txt.lower() == "q":
        break
    lab, conf = clf.predict(txt)
    print(f"→ {lab} ({conf})")

clf.save()  # writes artefacts to src/artifacts/
