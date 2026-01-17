import os
import joblib
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "sentiment_intensity_model_render"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR, "models", "tfidf_vectorizer.pkl"
)

# ✅ Render + Keras 3 safe loading
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

vectorizer = joblib.load(VECTORIZER_PATH)
