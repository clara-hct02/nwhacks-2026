chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "CLASSIFY") {
    fetch("http://127.0.0.1:8000/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg.message })
    })
      .then(res => res.json())
      .then(data => sendResponse(data))
      .catch(err => sendResponse({ error: err.toString() }));

    return true;
  }

  if (msg.type === "REASON") {
    fetch("http://127.0.0.1:8000/reason", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg.message })
    })
      .then(res => res.json())
      .then(data => sendResponse(data))
      .catch(err => sendResponse({ error: err.toString() }));

    return true;
  }
});
