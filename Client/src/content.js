const SCAM_PHRASES = [
  "send gift card",
  "send a gift card",
  "gift card urgently",
  "buy gift cards",
  "gift card asap"
];

function isScam(text) {
  const lower = text.toLowerCase();
  return SCAM_PHRASES.some(p => lower.includes(p));
}

function scanTextNodes(root = document.body) {
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        if (!node.parentElement) return NodeFilter.FILTER_REJECT;
        if (node.parentElement.closest(".scam-highlight")) {
          return NodeFilter.FILTER_REJECT;
        }
        if (node.textContent.trim().length < 10) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    }
  );

  let node;
  while ((node = walker.nextNode())) {
    if (isScam(node.textContent)) {
      highlightNode(node);
    }
  }
}

function highlightNode(textNode) {
  const span = document.createElement("span");
  span.className = "scam-highlight";
  span.textContent = textNode.textContent;

  const badge = document.createElement("span");
  badge.textContent = " ⚠️ SCAM";
  badge.className = "scam-badge";

  span.appendChild(badge);
  textNode.parentNode.replaceChild(span, textNode);
}

// Initial scan
scanTextNodes();

// Observe dynamic updates (chat apps, email, etc.)
const observer = new MutationObserver(mutations => {
  for (const m of mutations) {
    m.addedNodes.forEach(node => {
      if (node.nodeType === Node.ELEMENT_NODE) {
        scanTextNodes(node);
      }
    });
  }
});

observer.observe(document.body, {
  childList: true,
  subtree: true
});
