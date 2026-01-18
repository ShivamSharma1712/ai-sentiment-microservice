from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.model_loader import model, vectorizer
from nlp.text_cleaning import clean_text
from nlp.preprocessing import detect_flags


# -------------------------------
# VADER Sentiment Analyzer
# -------------------------------
vader = SentimentIntensityAnalyzer()


# -------------------------------
# Main Prediction Function
# -------------------------------
def predict_sentiment(text: str):
    """
    Sentiment prediction using:
    - ANN (TF-IDF + ANN)
    - VADER
    - Rule-based linguistic flags
    """

    # ---- ANN Prediction ----
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned]).toarray()
    ann_score = float(model.predict(vec, verbose=0)[0][0])

    # ---- VADER Prediction ----
    vader_score = vader.polarity_scores(text)

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
        "flags": flags
    }
