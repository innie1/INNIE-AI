/**
 * app.js
 * ------
 * Frontend logic for the INNIE AI chat interface.
 * Talks to the local Flask API defined in backend/api.py.
 */

const API_BASE = "http://127.0.0.1:5050/api";

const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const statusDot = document.getElementById("status-dot");

/** Append a chat bubble to the chat window and scroll to the bottom. */
function addMessage(role, text) {
  const messageEl = document.createElement("div");
  messageEl.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  messageEl.appendChild(bubble);
  chatWindow.appendChild(messageEl);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

/** Ping the API's health endpoint to show an online/offline status dot. */
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    statusDot.classList.toggle("online", true);
    statusDot.classList.toggle("offline", false);
    statusDot.title = data.trained
      ? "Connected -- model is trained"
      : "Connected -- model NOT trained yet (run trainer.py)";
  } catch (err) {
    statusDot.classList.toggle("online", false);
    statusDot.classList.toggle("offline", true);
    statusDot.title = "Cannot reach INNIE AI API on http://127.0.0.1:5050";
  }
}

/** Send the user's message to the backend and render the reply. */
async function sendMessage(message) {
  addMessage("user", message);

  const sendButton = chatForm.querySelector("button");
  sendButton.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await res.json();

    if (!res.ok) {
      addMessage("assistant", `Error: ${data.error || "something went wrong."}`);
      return;
    }

    addMessage("assistant", data.reply);
  } catch (err) {
    addMessage(
      "assistant",
      "Could not reach the INNIE AI API. Is backend/api.py running on port 5050?"
    );
  } finally {
    sendButton.disabled = false;
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  chatInput.value = "";
  sendMessage(message);
});

// Check connection status on load, and periodically afterwards
checkHealth();
setInterval(checkHealth, 15000);
