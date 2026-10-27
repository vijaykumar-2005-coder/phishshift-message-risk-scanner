from pathlib import Path

import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "message_model.pkl"

model = joblib.load(MODEL_PATH)


def analyze_message(message):
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = list(model.classes_)
    scores = dict(zip(classes, probabilities))

    confidence = float(scores[prediction])

    if prediction == "phishing":
        result = "Potential Risk"
        explanation = (
            "The trained model classified this message as phishing. "
            "Verify it through an independently accessed official channel."
        )
    else:
        result = "No Obvious Risk Cues"
        explanation = (
            "The model classified this message as legitimate, "
            "but this does not guarantee that it is safe."
        )

    return {
        "result": result,
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "explanation": explanation,
        "risk_cues": [],
        "prototype": False,
        "model_type": "TF-IDF + Logistic Regression",
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please enter a message."}), 400

    if len(message) > 10000:
        return jsonify({"error": "Message is too long."}), 400

    return jsonify(analyze_message(message))


if __name__ == "__main__":
    app.run(debug=True)