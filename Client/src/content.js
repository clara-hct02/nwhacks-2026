const RED_FLAGS = [
  "send gift card", "buy gift cards", "gift card asap", "western union", 
  "send money", "telegram", "whatsapp", "crypto", "bitcoin"
];

const YELLOW_FLAGS = [
  "urgent", "kindly", "verify your account", "secret", "don't tell anyone",
  "log in here", "password", "security code"
];

function checkThreatLevel(text) {
  const lower = text.toLowerCase();
  if (RED_FLAGS.some(p => lower.includes(p))) return "RED";
  if (YELLOW_FLAGS.some(p => lower.includes(p))) return "YELLOW";
  return null;
}

// Function to create the intrusive popup
function showWatchdogPopup(level, text) {
  // Check if a popup is already showing to avoid spamming the user
  if (document.getElementById('watchdog-alert-root')) return;

  const host = document.createElement('div');
  host.id = 'watchdog-alert-root';
  const shadow = host.attachShadow({ mode: 'open' });

  const isRed = level === "RED";
  const bgColor = isRed ? "#FF0000" : "#FFD700"; // Red or Yellow
  const textColor = isRed ? "#FFFFFF" : "#000000"; // White text on Red, Black on Yellow
  const warningText = isRed ? "STOP: DO NOT SEND MONEY OR GIFT CARDS." : "CAUTION: SUSPICIOUS CONVERSATION.";

  shadow.innerHTML = `
    <style>
      .overlay {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0,0,0,0.7);
        display: flex; align-items: center; justify-content: center;
        z-index: 2147483647;
        font-family: 'Segoe UI', Arial, sans-serif;
      }
      .modal {
        background: white;
        width: 90%; max-width: 500px;
        border: 10px solid ${bgColor};
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        animation: pop 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
      }
      @keyframes pop { from { transform: scale(0.8); opacity: 0; } to { transform: scale(1); opacity: 1; } }
      .header {
        background: ${bgColor};
        color: ${textColor};
        padding: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
      }
      .content {
        padding: 30px;
        color: #000;
        text-align: center;
        font-size: 18px;
        line-height: 1.5;
      }
      .btn {
        display: block;
        width: 80%;
        margin: 0 auto 20px auto;
        padding: 15px;
        background: #000;
        color: #fff;
        text-decoration: none;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        cursor: pointer;
        border: none;
        font-size: 16px;
      }
      .btn:hover { background: #333; }
    </style>
    <div class="overlay">
      <div class="modal">
        <div class="header">⚠️ ${warningText}</div>
        <div class="content">
          Watchdog detected <strong>suspicious behavior</strong> in this message. 
          Scammers often use phrases like this to steal money or information.
        </div>
        <button class="btn" id="close-btn">I am safe, close this</button>
      </div>
    </div>
  `;

  document.body.appendChild(host);

  shadow.getElementById('close-btn').addEventListener('click', () => {
    host.remove();
  });
}

function scanTextNodes(root = document.body) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
  let node;
  while ((node = walker.nextNode())) {
    const level = checkThreatLevel(node.textContent);
    if (level) {
      showWatchdogPopup(level, node.textContent);
      highlightNode(node, level);
    }
  }
}

function highlightNode(textNode, level) {
  if (textNode.parentNode.classList?.contains("scam-highlight")) return;

  const span = document.createElement("span");
  span.className = `scam-highlight ${level.toLowerCase()}`;
  span.textContent = textNode.textContent;

  const badge = document.createElement("span");
  badge.textContent = " ⚠️ WATCHDOG ALERT";
  badge.className = "scam-badge";

  span.appendChild(badge);
  textNode.parentNode.replaceChild(span, textNode);
}

// Initial scan and Observer
scanTextNodes();
const observer = new MutationObserver(mutations => {
  for (const m of mutations) {
    m.addedNodes.forEach(node => {
      if (node.nodeType === Node.ELEMENT_NODE) scanTextNodes(node);
    });
  }
});
observer.observe(document.body, { childList: true, subtree: true });