"""
api.py
------
A small local Flask API that lets frontend/chat.html talk to Brain.

Endpoints:
    GET  /api/health          -> simple status check
    POST /api/chat            -> {"message": str} -> {"reply": str}
    GET  /api/memory          -> list long-term memory facts
    POST /api/memory          -> {"fact": str} -> remember a new fact

Run with:  python api.py
Then open frontend/chat.html in a browser.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from config import API_HOST, API_PORT, DEBUG_MODE
from brain import Brain

app = Flask(__name__)
CORS(app)  # allow the frontend (opened as a local file / different port) to call this API

# One shared Brain instance for the lifetime of the server process
brain = Brain()


@app.route("/api/health", methods=["GET"])
def health():
    """Simple endpoint the frontend can ping to confirm the API is alive."""
    return jsonify({
        "status": "ok",
        "trained": brain.is_trained(),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint: takes a user message, returns INNIE AI's reply."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Field 'message' is required and cannot be empty."}), 400

    reply = brain.think(message)
    return jsonify({"reply": reply})


@app.route("/api/memory", methods=["GET"])
def get_memory():
    """Return everything currently stored in long-term memory."""
    return jsonify({"facts": brain.recall_facts()})


@app.route("/api/memory", methods=["POST"])
def post_memory():
    """Add a new fact to long-term memory."""
    data = request.get_json(silent=True) or {}
    fact = data.get("fact", "").strip()

    if not fact:
        return jsonify({"error": "Field 'fact' is required and cannot be empty."}), 400

    brain.remember_fact(fact)
    return jsonify({"status": "remembered", "fact": fact})


if __name__ == "__main__":
    print(f"INNIE AI API starting on http://{API_HOST}:{API_PORT}")
    print(f"Model trained: {brain.is_trained()}")
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG_MODE)
