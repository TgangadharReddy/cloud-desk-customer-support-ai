/**
 * CloudDesk Support — chat UI logic.
 *
 * Talks to the existing FastAPI pipeline via POST /chat and renders the
 * full SupportResponse: answer text, category, confidence, escalation
 * status/reason, and retrieved-FAQ sources when present.
 *
 * This file does NOT contain any classification, retrieval, or LLM
 * logic itself -- all of that already happened on the backend by the
 * time a response reaches here. This is purely presentation + a single
 * network call.
 *
 * Security note: every dynamic string that comes from the backend
 * (response text, category, escalation_reason, source questions) is
 * assigned via .textContent onto a pre-built, static DOM node -- never
 * interpolated into an innerHTML string. innerHTML is only ever used
 * with markup this file itself wrote (no backend data inside it).
 */

// Same-origin by default when served via serve_with_ui.py (both API and
// UI on http://127.0.0.1:8000). Written as an absolute URL per the
// project spec, which also happens to match that same-origin setup.
const API_URL = "https://cloud-desk-customer-support-ai.onrender.com/chat";
const HEALTH_URL = "https://cloud-desk-customer-support-ai.onrender.com/health";

const CONNECTION_ERROR_MESSAGE =
  "Unable to connect to CloudDesk Support. Please check that the support service is running and try again.";

const CATEGORY_CLASS_MAP = {
  Billing: "category-billing",
  Technical: "category-technical",
  "Account Access": "category-account",
};

const chatEl = document.getElementById("chat");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("messageInput");
const sendButtonEl = document.getElementById("sendButton");

// Two live copies of the connection status (header = always visible,
// sidebar = desktop only) -- both updated together by checkBackendHealth().
const statusPairs = [
  { dot: document.getElementById("statusDot"), label: document.getElementById("statusLabel") },
  { dot: document.getElementById("sidebarStatusDot"), label: document.getElementById("sidebarStatusLabel") },
];

// Captured once at load time so "New Conversation" can restore the exact
// original welcome state later. This HTML originates entirely from our
// own index.html, never from backend/user data, so re-assigning it via
// innerHTML is safe.
const welcomeStateHtml = document.getElementById("welcomeState").outerHTML;

/** Scroll the transcript to the latest message. */
function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

/**
 * Grow the textarea to fit its content, up to the max-height set in CSS
 * (beyond that, CSS overflow-y:auto takes over with a normal scrollbar).
 */
function autoResizeTextarea() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${inputEl.scrollHeight}px`;
}

/** Remove the welcome/empty state the first time a real turn is rendered. */
function hideWelcomeState() {
  const welcome = document.getElementById("welcomeState");
  if (welcome) {
    welcome.remove();
  }
}

/** Render the customer's own message as a right-aligned bubble. */
function renderCustomerTurn(message) {
  hideWelcomeState();

  const turn = document.createElement("div");
  turn.className = "turn customer";

  const bubble = document.createElement("div");
  bubble.className = "customer-bubble";
  bubble.textContent = message; // backend/user text -> textContent, never innerHTML

  turn.appendChild(bubble);
  chatEl.appendChild(turn);
  scrollToBottom();
}

/**
 * Render a temporary "AI is thinking" card while waiting on the backend.
 * Returns the element so the caller can remove it later.
 */
function renderTypingIndicator() {
  const turn = document.createElement("div");
  turn.className = "turn assistant";
  turn.innerHTML = `
    <span class="sender-label">CloudDesk AI</span>
    <div class="typing-card">
      <span>AI is thinking</span>
      <span class="typing-dots"><span></span><span></span><span></span></span>
    </div>
  `;
  chatEl.appendChild(turn);
  scrollToBottom();
  return turn;
}

/** Render a plain error card when the request itself failed (network/HTTP error). */
function renderErrorTurn(message) {
  const turn = document.createElement("div");
  turn.className = "turn assistant";

  const label = document.createElement("span");
  label.className = "sender-label";
  label.textContent = "CloudDesk AI";

  const card = document.createElement("div");
  card.className = "error-card";
  card.textContent = message;

  turn.appendChild(label);
  turn.appendChild(card);
  chatEl.appendChild(turn);
  scrollToBottom();
}

/** Build the escalation warning card. All backend text set via textContent. */
function buildEscalationCard(data) {
  const card = document.createElement("div");
  card.className = "escalation-card";
  card.innerHTML = `
    <div class="escalation-head">
      <svg viewBox="0 0 20 20" fill="none"><path d="M10 3 2 17h16L10 3Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="M10 8v3.2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="10" cy="14" r="0.9" fill="currentColor"/></svg>
      <span>Human Support Required</span>
    </div>
    <p class="escalation-body"></p>
    <div class="escalation-reason">
      <span class="reason-label">Reason</span>
      <span class="reason-text"></span>
    </div>
  `;
  card.querySelector(".escalation-body").textContent = data.response;
  card.querySelector(".reason-text").textContent =
    data.escalation_reason || "This request needs review.";
  return card;
}

/** Build the normal grounded-answer card: response + category/confidence + sources. */
function buildAiCard(data) {
  const card = document.createElement("div");
  card.className = "ai-card";
  card.innerHTML = `
    <p class="ai-response-text"></p>
    <div class="ai-meta">
      <span class="category-pill"></span>
      <div class="confidence-meter">
        <div class="confidence-track"><div class="confidence-fill"></div></div>
        <span class="confidence-value"></span>
      </div>
    </div>
  `;

  card.querySelector(".ai-response-text").textContent = data.response;

  const confidencePct = Math.round((data.confidence || 0) * 100);
  const categoryClass = CATEGORY_CLASS_MAP[data.category] || "category-unknown";
  const pill = card.querySelector(".category-pill");
  pill.classList.add(categoryClass);
  pill.textContent = data.category || "Unknown";

  let confidenceLevel = "confidence-low";
  if (confidencePct >= 70) confidenceLevel = "confidence-high";
  else if (confidencePct >= 40) confidenceLevel = "confidence-medium";

  const fill = card.querySelector(".confidence-fill");
  fill.classList.add(confidenceLevel);
  fill.style.width = `${confidencePct}%`;
  card.querySelector(".confidence-value").textContent = `${confidencePct}%`;

  if (Array.isArray(data.retrieved_faqs) && data.retrieved_faqs.length > 0) {
    const sourcesWrap = document.createElement("div");
    sourcesWrap.className = "ai-sources";

    const label = document.createElement("div");
    label.className = "sources-label";
    label.textContent = "Knowledge sources";
    sourcesWrap.appendChild(label);

    data.retrieved_faqs.slice(0, 3).forEach((faq) => {
      const chip = document.createElement("div");
      chip.className = "source-chip";

      const q = document.createElement("span");
      q.className = "source-question";
      q.textContent = faq.question;
      q.title = faq.question;

      const score = document.createElement("span");
      score.className = "source-score";
      const scorePct = Math.round((faq.score || 0) * 100);
      score.textContent = `${scorePct}%`;

      chip.appendChild(q);
      chip.appendChild(score);
      sourcesWrap.appendChild(chip);
    });

    card.appendChild(sourcesWrap);
  }

  return card;
}

/** Render the assistant's full SupportResponse, as either an escalation or a normal AI card. */
function renderAssistantTurn(data) {
  const turn = document.createElement("div");
  turn.className = "turn assistant";

  const label = document.createElement("span");
  label.className = "sender-label";
  label.textContent = "CloudDesk AI";
  turn.appendChild(label);

  const card = data.escalated ? buildEscalationCard(data) : buildAiCard(data);
  turn.appendChild(card);

  chatEl.appendChild(turn);
  scrollToBottom();
}

/** POST a message to the backend and return the parsed SupportResponse. */
async function sendMessage(message) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error(`Server responded with status ${response.status}`);
  }

  return response.json();
}

/** Send a given message through the full chat flow (used by both the composer and suggestion chips). */
async function submitMessage(message) {
  const trimmed = message.trim();
  if (!trimmed) {
    return;
  }

  renderCustomerTurn(trimmed);
  inputEl.value = "";
  autoResizeTextarea();
  inputEl.disabled = true;
  sendButtonEl.disabled = true;

  const typingTurn = renderTypingIndicator();

  try {
    const data = await sendMessage(trimmed);
    typingTurn.remove();
    renderAssistantTurn(data);
  } catch (err) {
    typingTurn.remove();
    renderErrorTurn(CONNECTION_ERROR_MESSAGE);
    console.error("Chat request failed:", err);
  } finally {
    inputEl.disabled = false;
    sendButtonEl.disabled = false;
    inputEl.focus();
  }
}

/** Handle the composer form submission. */
function handleSubmit(event) {
  event.preventDefault();
  submitMessage(inputEl.value);
}

/**
 * Enter sends the message; Shift+Enter inserts a normal newline instead.
 * Submits via the form's own submit event so there is one single code
 * path for "send a message" regardless of what triggered it.
 */
function handleComposerKeydown(event) {
  if (event.key !== "Enter" || event.shiftKey) {
    return;
  }
  event.preventDefault();
  if (sendButtonEl.disabled) {
    return;
  }
  if (typeof formEl.requestSubmit === "function") {
    formEl.requestSubmit();
  } else {
    formEl.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
  }
}

/** Event delegation for welcome-state suggestion chips (survives resetConversation's innerHTML swap). */
function handleChatClick(event) {
  const chip = event.target.closest(".suggestion-chip");
  if (chip && chip.dataset.message) {
    submitMessage(chip.dataset.message);
  }
}

/** Clear the transcript and restore the original welcome state. Frontend-only; backend is stateless per request. */
function resetConversation() {
  chatEl.innerHTML = welcomeStateHtml;
  inputEl.value = "";
  autoResizeTextarea();
  inputEl.focus();
}

/** Ping /health on load to show a connection status indicator (both header and sidebar copies). */
async function checkBackendHealth() {
  try {
    const response = await fetch(HEALTH_URL);
    if (!response.ok) throw new Error("Unhealthy");
    statusPairs.forEach(({ dot, label }) => {
      if (!dot || !label) return;
      dot.classList.remove("offline");
      dot.classList.add("online");
      label.textContent = "Connected";
    });
  } catch (err) {
    statusPairs.forEach(({ dot, label }) => {
      if (!dot || !label) return;
      dot.classList.remove("online");
      dot.classList.add("offline");
      label.textContent = "Backend unavailable";
    });
  }
}

formEl.addEventListener("submit", handleSubmit);
inputEl.addEventListener("input", autoResizeTextarea);
inputEl.addEventListener("keydown", handleComposerKeydown);
chatEl.addEventListener("click", handleChatClick);
document.querySelectorAll(".new-convo-trigger").forEach((btn) => {
  btn.addEventListener("click", resetConversation);
});

checkBackendHealth();
inputEl.focus();