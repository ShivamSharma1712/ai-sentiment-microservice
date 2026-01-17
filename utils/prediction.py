import os
import requests

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.model_loader import model, vectorizer
from nlp.text_cleaning import clean_text
from nlp.preprocessing import detect_flags


# -------------------------------
# VADER Sentiment Analyzer
# -------------------------------
vader = SentimentIntensityAnalyzer()


# -------------------------------
# Hugging Face Inference API
# -------------------------------
HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}


def hf_api_sentiment(text: str):
    """
    Calls Hugging Face hosted inference API
    instead of loading transformer locally.
    """

    try:
        response = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={"inputs": text},
            timeout=15
        )

        if response.status_code != 200:
            return {
                "label": "ERROR",
                "score": 0.0
            }

        result = response.json()

        # Expected format: [[{'label': 'POSITIVE', 'score': 0.99}]]
        prediction = result[0][0]

        return {
            "label": prediction.get("label"),
            "score": round(float(prediction.get("score", 0.0)), 4)
        }

    except Exception:
        return {
            "label": "ERROR",
            "score": 0.0
        }


# -------------------------------
# Main Prediction Function
# -------------------------------
def predict_sentiment(text: str):
    """
    Hybrid sentiment prediction:
    - ANN (TF-IDF + ANN)
    - VADER
    - Hugging Face (API-based)
    - Rule-based flags
    """

    # ---- ANN Prediction ----
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned]).toarray()
    ann_score = float(model.predict(vec, verbose=0)[0][0])

    # ---- VADER Prediction ----
    vader_score = vader.polarity_scores(text)

    # ---- Hugging Face Prediction (API) ----
    hf_result = hf_api_sentiment(text)

    # ---- Linguistic Flags ----
    flags = detect_flags(text)

    return {
        "custom_ann": {
            "score": round(ann_score, 2),
            "confidence_range": [
                round(ann_score - 0.4, 2),
                round(ann_score + 0.4, 2)
            ]
        },
        "vader": vader_score,
        "huggingface": hf_result,
        "flags": flags
    }
