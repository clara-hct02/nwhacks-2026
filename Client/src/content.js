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

function getScamAnalysis(text) {
  const lower = text.toLowerCase();
  for (const phrase in SCAM_DB) {
    if (lower.includes(phrase)) {
      return { phrase, ...SCAM_DB[phrase] };
    }
  }
  return null;
}

// 1. Create Global Portal for the "Mini-Popup"
const miniPopup = document.createElement('div');
miniPopup.id = 'watchdog-mini-popup';
miniPopup.style.display = 'none';
document.body.appendChild(miniPopup);

function highlightNode(textNode, analysis) {
  const span = document.createElement("span");
  span.className = `scam-highlight ${analysis.level.toLowerCase()}`;
  
  const badge = document.createElement("span");
  badge.className = "scam-badge";
  badge.textContent = "⚠️ ALERT";

  const handleAlertClick = (e) => {
    e.stopPropagation();
    
    const rect = badge.getBoundingClientRect();
    const threatLabel = analysis.level === "RED" ? "HIGH" : "LOW";
    const themeColor = analysis.level === "RED" ? "#e92d2d" : "#f4bd3c";

    miniPopup.innerHTML = `
      <div style="border-top: 5px solid ${themeColor}; padding: 10px;">
        <div style="font-weight: bold; margin-bottom: 8px;">
          Labelled <span style="color: ${themeColor}">${threatLabel}</span> threat
        </div>
        <button id="wd-learn-more" class="mini-btn">Learn More</button>
      </div>
    `;

    miniPopup.style.display = 'block';
    miniPopup.style.top = `${rect.top + window.scrollY - 10}px`;
    miniPopup.style.left = `${rect.left + window.scrollX + (rect.width / 2)}px`;

    document.getElementById('wd-learn-more').onclick = async () => {
      miniPopup.style.display = 'none';

      const server = await new Promise((resolve) => {
        chrome.runtime.sendMessage(
          { type: "REASON", message: textNode.textContent },
          (response) => resolve(response)
        );
      });

      const reason = server?.reasoning ?? "Unable to load explanation.";
      showWatchdogPopup(analysis.level, reason);
    };
  };

  badge.onclick = handleAlertClick;
  span.onclick = handleAlertClick;

  span.textContent = textNode.textContent;
  span.appendChild(badge);

  if (textNode.parentNode) {
    textNode.parentNode.replaceChild(span, textNode);
  }
}

document.addEventListener('mousedown', (e) => {
  if (miniPopup.style.display === 'block' && 
      !miniPopup.contains(e.target) && 
      !e.target.closest('.scam-highlight')) {
    miniPopup.style.display = 'none';
  }
});

function showWatchdogPopup(level, reason) {
  if (document.getElementById('watchdog-alert-root')) return;
  
  const host = document.createElement('div');
  host.id = 'watchdog-alert-root';
  const shadow = host.attachShadow({ mode: 'open' });

  const isRed = level === "RED";

  const mascotUrl = chrome.runtime.getURL(
    isRed ? "likelyscam.png" : "potentialscam.png"
  );

  const themeColor = isRed ? "#c41428" : "#f4bd3c";


  shadow.innerHTML = `
    <style>
      .overlay { 
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; 
        background: rgba(0,0,0,0.7); display: flex; align-items: center; 
        justify-content: center; z-index: 2147483647; font-family: 'Arial', sans-serif; 
      }

.modal { 
  background: #fff;
  width: 90%;
  max-width: 400px;
  border: 10px solid ${themeColor}; 
  border-radius: 35px;
  box-shadow: 0 20px 50px rgba(0,0,0,0.5); 
  position: relative;

  display: flex;
  flex-direction: column;
  align-items: center;

  /* Move content closer to top */
  padding: 80px 20px 25px; /* previously 110px */
}

      .mascot-container {
        position: absolute;
        top: -90px;          /* ⬅ pull mascot above card */
        left: 50%;
        transform: translateX(-50%);
        z-index: 2;
      }

      .header { 
        background: ${themeColor}; 
        color: white; 
        padding: 15px; 
        font-weight: bold; 
        font-size: 26px; 
        letter-spacing: 2px;
      }
      .mascot-img {
        width: 400px;
        height: auto;
      }
.content { 
  /* Center text horizontally */
  text-align: center;

  /* Pull text upward */
  margin-top: 0;

  /* Keep full width so centering works cleanly */
  align-self: stretch;

  color: #000; 
  font-size: 18px; 
  line-height: 1.4; 
  margin-bottom: 20px;
}
.btn { 
  display: inline-block;
  background: #000; 
  color: #fff; 
  padding: 12px 10px; 
  border-radius: 50px; 
  font-weight: bold; 
  cursor: pointer; 
  border: none; 

  /* NEW: smaller text + more spacing */
  font-size: 14px;
  letter-spacing: 2px;

  transition: transform 0.1s;
}

      .btn:hover { background: ${themeColor}; }
      .btn:active { transform: scale(0.95); }
    </style>
    <div class="overlay">
    <div class="modal">
      <div class="mascot-container">
        <img src="${mascotUrl}" class="mascot-img" alt="Watchdog Mascot">
      </div>

      <div class="content">
        ${reason}
      </div>

      <button class="btn" id="close-btn">I UNDERSTAND</button>
    </div>

    </div>
  `;
  document.body.appendChild(host);
  shadow.getElementById('close-btn').onclick = () => host.remove();
}

// ---------------------------------------------------------
// Messenger-only message scanning (bubble-based)
// ---------------------------------------------------------

async function scanTextNodes() {
  // Your real message bubbles look like:
  // <div dir="auto" class="html-div ...">send gift card</div>
  const messageNodes = document.querySelectorAll('div[dir="auto"].html-div');

  for (const node of messageNodes) {
    // Avoid scanning the same message twice
    if (node.classList.contains("wd-scanned")) continue;
    node.classList.add("wd-scanned");

    const text = node.innerText.trim();
    if (!text) continue;

    const result = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { type: "CLASSIFY", message: text },
        (response) => resolve(response)
      );
    });

    if (result && (result.threatLevel === "RED" || result.threatLevel === "YELLOW")) {
      if (node.firstChild) {
        highlightNode(node.firstChild, {
          level: result.threatLevel,
          reason: "Loading…"
        });
      }
    }
  }
}

// ---------------------------------------------------------
// Continuous scanning via MutationObserver
// ---------------------------------------------------------

let timeout;
const observer = new MutationObserver(() => {
  clearTimeout(timeout);
  timeout = setTimeout(() => {
    scanTextNodes();
  }, 200);
});

observer.observe(document.body, {
  childList: true,
  subtree: true
});

// Initial scan
scanTextNodes();
