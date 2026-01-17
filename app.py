import os
import logging
from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api, Resource, fields

from utils.prediction import predict_sentiment

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    filename="requests.log",           # log file
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# -------------------------------
# Flask App Setup
# -------------------------------
app = Flask(__name__)
CORS(app)

api = Api(
    app,
    title="AI Sentiment Analysis Microservice",
    description="ANN + HuggingFace + VADER based sentiment analysis",
    doc="/docs"   # Swagger UI URL
)

ns = api.namespace("sentiment", description="Sentiment Operations")

input_model = api.model("InputText", {
    "text": fields.String(required=True, description="Customer feedback text")
})

# -------------------------------
# Health Endpoint
# -------------------------------
@ns.route("/health")
class Health(Resource):
    def get(self):
        logging.info("/health | status=200")
        return {
            "service": "AI Sentiment Microservice",
            "status": "running",
            "models_loaded": True
        }

# -------------------------------
# Prediction Endpoint
# -------------------------------
@ns.route("/predict")
class Predict(Resource):
    @ns.expect(input_model)
    def post(self):
        data = request.json
        text = data.get("text", "").strip()

        if not text:
            logging.info("/predict | EMPTY_INPUT | status=400")
            return {"error": "Empty input"}, 400

        result = predict_sentiment(text)

        logging.info(
            f"/predict | text_length={len(text)} | status=200"
        )

        return result, 200

# -------------------------------
# App Runner
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
