# ================================
# Customer Sentiment Analysis
# Keras 3 – Local Stable Training
# Exam-safe | Deployment-ready
# ================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

import joblib

# -------------------------------
# 1. Setup
# -------------------------------
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

DATA_PATH = "amazon_reviews.csv"   # keep file in project root
# MODEL_OUT = "sentiment_intensity_model_v3.keras"
# VECT_OUT  = "tfidf_vectorizer.pkl"


# -------------------------------
# 2. Load Dataset
# -------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("amazon_reviews.csv not found in project root")

df = pd.read_csv(DATA_PATH)
df = df[["Score", "Summary", "Text"]].dropna()

df["review"] = df["Summary"] + " " + df["Text"]

# -------------------------------
# 3. Text Cleaning
# -------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)

df["clean_review"] = df["review"].apply(clean_text)

# -------------------------------
# 4. TF-IDF (SPARSE – MEMORY SAFE)
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=3000,        # safe for local RAM
    ngram_range=(1, 2),
    min_df=5,
    dtype=np.float32
)

X_sparse = vectorizer.fit_transform(df["clean_review"])
y = df["Score"].astype(float)

# -------------------------------
# 5. Train-Test Split (SPARSE)
# -------------------------------
X_train_sp, X_test_sp, y_train, y_test = train_test_split(
    X_sparse, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Convert ONLY splits to dense
# -------------------------------
X_train = X_train_sp.toarray()
X_test = X_test_sp.toarray()

# -------------------------------
# 7. Keras 3–Safe ANN Architecture
# -------------------------------
model = Sequential([
    Input(shape=(X_train.shape[1],)),   # ✅ CRITICAL
    Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1, activation="linear")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# -------------------------------
# 8. Training
# -------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)


# -------------------------------
# 9. Evaluation (MEMORY SAFE)
# -------------------------------

y_pred = []
BATCH_SIZE = 512

for i in range(0, X_test_sp.shape[0], BATCH_SIZE):
    batch = X_test_sp[i:i+BATCH_SIZE].toarray()
    preds = model.predict(batch, verbose=0).flatten()
    y_pred.extend(preds)

y_pred = np.array(y_pred)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nMAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")


# -------------------------------
# 10. Save Artifacts (IMPORTANT)
# -------------------------------
# -------------------------------
# 10. Save Artifacts (RENDER SAFE)
# -------------------------------

MODEL_DIR = "sentiment_intensity_model_render"

if os.path.exists(MODEL_DIR):
    import shutil
    shutil.rmtree(MODEL_DIR)

tf.saved_model.save(model, MODEL_DIR)

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nMODEL SAVED SUCCESSFULLY IN SAVEDMODEL FORMAT")
print("MODEL DIR CONTENTS:", os.listdir(MODEL_DIR))


# -------------------------------
# 11. Visualization (NON-BLOCKING)
# -------------------------------
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Sentiment Intensity Regression")
plt.savefig("training_scatter.png")
plt.close()


# -------------------------------
# 12. Negation Analysis (REPORT)
# -------------------------------
NEGATIONS = {"not", "no", "never", "hardly", "barely"}
POSITIVE = {"good", "great", "excellent", "amazing", "nice"}
NEGATIVE = {"bad", "worst", "awful", "terrible"}

def detect_negation_issue(text: str):
    tokens = text.lower().split()
    for i in range(len(tokens) - 1):
        if tokens[i] in NEGATIONS and tokens[i + 1] in POSITIVE:
            return "NEGATION_OF_POSITIVE"
        if tokens[i] in NEGATIONS and tokens[i + 1] in NEGATIVE:
            return "NEGATION_OF_NEGATIVE"
    return None

print("\nSample Negation Analysis:")
for i in range(3):
    raw = df.iloc[i]["review"]
    print("-", detect_negation_issue(raw))
