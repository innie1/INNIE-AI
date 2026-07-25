"""
api.py
------
Flask API with per-request latency, throughput logging, memory, health, and live dashboard routes.
"""
import time
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from config import API_HOST, API_PORT, DEBUG_MODE, PERFORMANCE_LOG_PATH, FRONTEND_DIR
from brain import Brain
from performance import PerformanceMonitor

app = Flask(__name__)
CORS(app)

brain = Brain()
perf = PerformanceMonitor(log_path=PERFORMANCE_LOG_PATH)


@app.before_request
def before_request():
    """Attach start time to request context."""
    request._start_time = time.perf_counter()


@app.after_request
def after_request(response):
    """Log request latency and status."""
    if hasattr(request, "_start_time"):
        elapsed = time.perf_counter() - request._start_time
        perf.log("api_request", {
            "method": request.method,
            "endpoint": request.endpoint,
            "path": request.path,
            "status_code": response.status_code,
            "latency_sec": elapsed,
            "content_length": response.content_length or 0,
        })
    return response


@app.route("/api/health", methods=["GET"])
def health():
    """Health check with full system snapshot including GPU, RAM, CPU."""
    snapshot = perf.get_system_snapshot()
    return jsonify({
        "status": "ok",
        "trained": brain.is_trained(),
        **snapshot,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint with performance metadata."""
    perf.tick("api_chat_total")

    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    max_tokens = data.get("max_tokens", 50)
    temperature = data.get("temperature", 1.0)

    if not user_message:
        return jsonify({"error": "Field 'message' is required and cannot be empty."}), 400

    perf.tick("brain_think")
    result = brain.think_with_metrics(user_message, max_new_tokens=max_tokens, temperature=temperature)
    brain_time = perf.tock("brain_think")

    total_time = perf.tock("api_chat_total")

    reply = result.get("text", "")
    response_payload = {
        "reply": reply,
        "response": reply,
        "generated": result.get("generated", reply),
        "performance": {
            "api_total_sec": total_time,
            "brain_think_sec": brain_time,
            **result.get("metrics", {}),
        }
    }

    perf.log("api_chat_complete", {
        "prompt_length": len(user_message),
        "max_tokens": max_tokens,
        "api_total_sec": total_time,
        "brain_think_sec": brain_time,
    })

    return jsonify(response_payload)


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


@app.route("/api/performance", methods=["GET"])
def performance_report():
    """Return aggregated performance statistics."""
    event_filter = request.args.get("event")
    summary = perf.summarize(event_filter=event_filter or None)
    return jsonify({
        "event_filter": event_filter,
        "summary": summary,
        "log_path": PERFORMANCE_LOG_PATH,
    })


@app.route("/api/performance/summary", methods=["GET"])
def performance_summary_text():
    """Return a plain-text performance summary."""
    from io import StringIO
    import sys

    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    perf.print_summary()
    sys.stdout = old_stdout

    return buffer.getvalue(), 200, {"Content-Type": "text/plain"}


@app.route("/api/performance/live", methods=["GET"])
def performance_live():
    """Return recent events for live dashboard."""
    events = perf.load_history()
    recent = events[-100:] if len(events) > 100 else events
    return jsonify({
        "events": list(reversed(recent)),
        "total_events": len(events),
    })


# ── Serve frontend files ──────────────────────────────────────────

@app.route("/")
def serve_chat():
    if os.path.exists(os.path.join(FRONTEND_DIR, "chat.html")):
        return send_from_directory(FRONTEND_DIR, "chat.html")
    return jsonify({"message": "INNIE AI API is running"}), 200


@app.route("/dashboard")
def serve_dashboard():
    if os.path.exists(os.path.join(FRONTEND_DIR, "dashboard.html")):
        return send_from_directory(FRONTEND_DIR, "dashboard.html")
    return jsonify({"error": "dashboard.html not found"}), 404


@app.route("/<path:filename>")
def serve_static(filename):
    if os.path.exists(os.path.join(FRONTEND_DIR, filename)):
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({"error": f"File {filename} not found"}), 404


if __name__ == "__main__":
    print(f"Starting INNIE AI API on http://{API_HOST}:{API_PORT}")
    print(f"Chat UI:     http://{API_HOST}:{API_PORT}/")
    print(f"Dashboard:   http://{API_HOST}:{API_PORT}/dashboard")
    print(f"Performance logs -> {PERFORMANCE_LOG_PATH}")
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG_MODE)
