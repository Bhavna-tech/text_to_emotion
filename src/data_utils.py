
# from pathlib import Path
# import pandas as pd
# import neattext.functions as nfx

# def load_emotion_txt(path: Path) -> pd.DataFrame:
#     rows = []
#     with path.open(encoding="utf-8") as fh:
#         for raw in fh:
#             parts = raw.strip().split(";")
#             if len(parts) == 2:
#                 rows.append(parts)
#     return pd.DataFrame(rows, columns=["Text", "Emotion"])


# def normalise(text: str) -> str:
#     return nfx.remove_stopwords(nfx.remove_special_characters(text.lower()))
from pathlib import Path
import pandas as pd
import neattext.functions as nfx

def load_emotion_txt(path: Path) -> pd.DataFrame:
    """Read a `text;label` file into a DataFrame."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.strip().split(";")
            if len(parts) == 2:
                rows.append(parts)
    return pd.DataFrame(rows, columns=["Text", "Emotion"])

def normalise(text: str) -> str:
    """Lower-case, strip special chars & stop-words."""
    return nfx.remove_stopwords(nfx.remove_special_characters(text.lower()))
