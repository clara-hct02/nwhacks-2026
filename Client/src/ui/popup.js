document.querySelector('.info-logo').addEventListener('click', async () => {
    // Get the current active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (tab) {
      // Send message to content.js
      chrome.tabs.sendMessage(tab.id, { action: "SHOW_WELCOME" });
      // Close the small popup window so the user sees the big one on the page
      window.close();
    }
  });