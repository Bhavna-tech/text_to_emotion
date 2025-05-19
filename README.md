# Text to Emotion Detection

This project detects emotions (joy, sadness, anger, etc.) from text using classical ML techniques (TF-IDF + Linear SVM).

##  Folder Structure
```
TEXT_TO_EMOTION/
├── data/ # Raw dataset (train/val/test)
├── notebooks/ # EDA and visualization
├── src/ # Codebase (data loading, model, features)
├── tests/ # tests file
├── run.py # Training and inference entrypoint
├── requirements.txt # Python package dependencies
├── .env # PYTHONPATH


## How to Run

1. Create virtual env:
   ```bash
   python -m venv venv 
2. Activate:

    Windows: venv\Scripts\activate

3. Install packages
    ```bash
   pip install -r requirements.txt
4. Run
    ```bash
    python run.py



