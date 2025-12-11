const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const output = document.querySelector("#output");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  appendMessage("user", "Tu", message);
  input.value = "";
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  const data = await resp.json();
  if (data.response) appendMessage("agent", "Tarara", data.response);
  if (data.error) appendMessage("agent", "Tarara", data.error);
});

function appendMessage(role, author, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<div class="author">${author}</div><div class="text">${text}</div>`;

  wrapper.appendChild(bubble);
  output.appendChild(wrapper);
  output.scrollTop = output.scrollHeight;
}
