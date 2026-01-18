// src/content.js

const SCAM_DB = {
  "send gift card": { level: "RED", reason: "Real companies or government agencies will NEVER ask you to pay them with gift cards." },
  "buy gift cards": { level: "RED", reason: "Scammers use gift cards because they are like cash—once you share the code, the money is gone forever." },
  "western union": { level: "RED", reason: "Wire transfers are a common way for scammers to take money without being tracked." },
  "crypto": { level: "RED", reason: "Investment scams often promise high returns via Cryptocurrency. These are almost always fraudulent." },
  "bitcoin": { level: "RED", reason: "Scammers prefer Bitcoin because it is difficult for banks to reverse or track." },
  "telegram": { level: "RED", reason: "Scammers often try to move you off Facebook to 'Telegram' or 'WhatsApp' to avoid security filters." },
  "whatsapp": { level: "RED", reason: "Moving a conversation to WhatsApp is a classic tactic to hide a scam from Facebook's detectors." },
  "urgent": { level: "YELLOW", reason: "Scammers create a 'sense of urgency' to make you panic and act without thinking." },
  "kindly": { level: "YELLOW", reason: "The word 'kindly' is a very common linguistic pattern used by overseas scammers." },
  "verify your account": { level: "YELLOW", reason: "Never click links to 'verify' your account. Go directly to the official website instead." },
  "secret": { level: "YELLOW", reason: "Scammers often tell you to keep things a secret from your family so they can't warn you." },
  "security code": { level: "YELLOW", reason: "Never share a security code sent to your phone. Scammers use these to hack into your accounts." }
};

// 1. Create the Global Tooltip ONCE
const globalTooltip = document.createElement('div');
globalTooltip.id = 'watchdog-global-tooltip';
document.body.appendChild(globalTooltip);

function fetchReasonFromServer(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(
      { type: "ANALYZE", message },
      (response) => {
        resolve(response?.reasoning ?? "No reasoning returned.");
      }
    );
  });
}

function getScamAnalysis(text) {
  const lower = text.toLowerCase();
  for (const phrase in SCAM_DB) {
    if (lower.includes(phrase)) {
      return { phrase, ...SCAM_DB[phrase] };
    }
  }
  return null;
}

function showWatchdogPopup(level, reason) {
  if (document.getElementById('watchdog-alert-root')) return;
  
  const host = document.createElement('div');
  host.id = 'watchdog-alert-root';
  const shadow = host.attachShadow({ mode: 'open' });

  // 1. Get the mascot image URL from your Client folder
  const mascotUrl = chrome.runtime.getURL('mascot.png');

  const isRed = level === "RED";
  const themeColor = isRed ? "#bc3e3e" : "#FFD700"; // Using the sketch's red

  shadow.innerHTML = `
    <style>
      .overlay { 
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; 
        background: rgba(0,0,0,0.7); display: flex; align-items: center; 
        justify-content: center; z-index: 2147483647; font-family: 'Arial', sans-serif; 
      }
      .modal { 
        background: #d9d9d9; /* Light grey background from your sketch */
        width: 90%; max-width: 400px; 
        border: 10px solid ${themeColor}; 
        border-radius: 35px; /* Hand-drawn rounded look */
        overflow: hidden; 
        box-shadow: 0 20px 50px rgba(0,0,0,0.5); 
        text-align: center;
        padding-bottom: 25px;
      }
      .header { 
        background: ${themeColor}; 
        color: white; 
        padding: 15px; 
        font-weight: bold; 
        font-size: 26px; 
        letter-spacing: 2px;
      }
      .mascot-container {
        padding: 20px 0;
      }
      .mascot-img {
        width: 140px; /* Adjust based on your PNG proportions */
        height: auto;
      }
      .content { 
        padding: 20px 30px; 
        color: #000; 
        font-size: 18px; 
        line-height: 1.4; 
        /* The horizontal lines from your sketch */
        border-top: 2px solid #999;
        border-bottom: 2px solid #999;
        margin: 0 30px 25px 30px;
      }
      .btn { 
        display: inline-block;
        background: #000; 
        color: #fff; 
        padding: 12px 40px; 
        border-radius: 50px; 
        font-weight: bold; 
        cursor: pointer; 
        border: none; 
        font-size: 18px; 
        transition: transform 0.1s;
      }
      .btn:hover { background: #222; }
      .btn:active { transform: scale(0.95); }
    </style>
    <div class="overlay">
      <div class="modal">
        <div class="header">⚠️ WARNING</div>
        
        <div class="mascot-container">
          <img src="${mascotUrl}" class="mascot-img" alt="Watchdog Mascot">
        </div>

        <div class="content">
          ${reason}
        </div>

        <button class="btn" id="close-btn">I Understand</button>
      </div>
    </div>
  `;
  document.body.appendChild(host);
  shadow.getElementById('close-btn').onclick = () => host.remove();
}

function scanTextNodes(root = document.body) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
  let node;
  const matches = [];

  while ((node = walker.nextNode())) {
    const parent = node.parentElement;
    if (!parent || ["SCRIPT", "STYLE", "TEXTAREA", "NOSCRIPT"].includes(parent.tagName)) continue;
    if (parent.closest(".scam-highlight")) continue;

    const analysis = getScamAnalysis(node.textContent);
    if (analysis) matches.push({ node, analysis });
  }

  matches.forEach(({ node, analysis }) => {
    highlightNode(node, analysis);

    if (analysis.level === "RED") {
      fetchReasonFromServer(node.textContent).then(serverReason => {
        showWatchdogPopup(analysis.level, serverReason);
      });
    }
  });
}

function highlightNode(textNode, analysis) {
  const span = document.createElement("span");
  span.className = `scam-highlight ${analysis.level.toLowerCase()}`;
  
  const badge = document.createElement("span");
  badge.className = "scam-badge";
  badge.textContent = "⚠️ ALERT";

  badge.onmouseenter = (e) => {
    const rect = badge.getBoundingClientRect();
    globalTooltip.innerHTML = `<strong>Why was this flagged?</strong><br>AI is analyzing this message...`;
    globalTooltip.style.display = 'block';
    globalTooltip.style.top = `${rect.top + window.scrollY - 10}px`;
    globalTooltip.style.left = `${rect.left + window.scrollX + (rect.width / 2)}px`;
  };

  badge.onmouseleave = () => {
    globalTooltip.style.display = 'none';
  };

  span.textContent = textNode.textContent;
  span.appendChild(badge);

  if (textNode.parentNode) {
    textNode.parentNode.replaceChild(span, textNode);
  }
}

// Initial Scan
scanTextNodes();

// Watch for dynamic changes (like Messenger loading or scrolling)
let timeout;
const observer = new MutationObserver(() => {
  clearTimeout(timeout);
  timeout = setTimeout(() => scanTextNodes(document.body), 800);
});

observer.observe(document.body, { childList: true, subtree: true });
