// Mimir Content Script - 注入 Claude 和 ChatGPT 页面
// 监听 DOM 变化，捕获对话内容

(function() {
  'use strict';

  // 检测当前平台
  const platform = window.location.hostname.includes('claude.ai') ? 'claude' : 'chatgpt';
  
  console.log(`[Mimir] Extension Loaded on ${platform}`);

  // 注入页面脚本以访问页面变量
  const script = document.createElement('script');
  script.src = chrome.runtime.getURL('inject.js');
  script.onload = function() {
    this.remove();
  };
  (document.head || document.documentElement).appendChild(script);

  // 与 background 通信的包装函数
  function sendToBackground(message) {
    return new Promise((resolve, reject) => {
      try {
        chrome.runtime.sendMessage(message, (response) => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve(response);
          }
        });
      } catch (error) {
        reject(error);
      }
    });
  }

  // 提取 Claude 对话
  function extractClaudeConversation() {
    const messages = [];
    // Claude 的对话选择器
    const messageElements = document.querySelectorAll('[data-testid="user-message"], [data-testid="assistant-message"]');
    
    messageElements.forEach((el, index) => {
      const isUser = el.getAttribute('data-testid') === 'user-message';
      const text = el.textContent?.trim();
      if (text) {
        messages.push({
          index,
          role: isUser ? 'user' : 'assistant',
          content: text,
          timestamp: Date.now()
        });
      }
    });
    
    return {
      platform: 'claude',
      url: window.location.href,
      title: document.title,
      messages,
      capturedAt: new Date().toISOString()
    };
  }

  // 提取 ChatGPT 对话
  function extractChatGPTConversation() {
    const messages = [];
    // ChatGPT 的对话选择器
    const messageElements = document.querySelectorAll('[data-message-author-role]');
    
    messageElements.forEach((el, index) => {
      const role = el.getAttribute('data-message-author-role');
      const textContent = el.querySelector('.markdown, .text-message')?.textContent?.trim() || 
                         el.textContent?.trim();
      if (textContent && role) {
        messages.push({
          index,
          role: role,
          content: textContent,
          timestamp: Date.now()
        });
      }
    });
    
    return {
      platform: 'chatgpt',
      url: window.location.href,
      title: document.title,
      messages,
      capturedAt: new Date().toISOString()
    };
  }

  // 提取对话的主函数
  function extractConversation() {
    if (platform === 'claude') {
      return extractClaudeConversation();
    } else {
      return extractChatGPTConversation();
    }
  }

  // 监听 DOM 变化
  let debounceTimer = null;
  const observer = new MutationObserver((mutations) => {
    // 防抖处理，避免频繁触发
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const conversation = extractConversation();
      if (conversation.messages.length > 0) {
        // 发送给 background
        sendToBackground({
          type: 'CONVERSATION_UPDATED',
          data: conversation
        }).catch(err => {
          console.log('[Mimir] Failed to send to background:', err.message);
        });
      }
    }, 2000); // 2秒防抖
  });

  // 开始监听
  function startObserving() {
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true
    });
    console.log('[Mimir] Started observing DOM changes');
  }

  // 页面加载完成后开始监听
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startObserving);
  } else {
    startObserving();
  }

  // 监听来自页面的消息（从 inject.js 传来的）
  window.addEventListener('message', (event) => {
    if (event.source !== window) return;
    
    if (event.data.type && event.data.type.startsWith('MIMIR_')) {
      // 转发给 background
      sendToBackground({
        type: event.data.type,
        data: event.data.data
      }).catch(err => {
        console.log('[Mimir] Failed to forward message:', err.message);
      });
    }
  });

  // 监听来自 background/popup 的消息
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('[Mimir] Received message:', request);
    
    switch (request.type) {
      case 'GET_CONVERSATION':
        const conversation = extractConversation();
        sendResponse({ success: true, data: conversation });
        break;
        
      case 'PING':
        sendResponse({ success: true, platform, message: 'Mimir Extension Loaded' });
        break;
        
      default:
        sendResponse({ success: false, error: 'Unknown message type' });
    }
    
    return true; // 保持消息通道开放
  });

  // 通知 background 页面已加载
  sendToBackground({
    type: 'PAGE_LOADED',
    data: { platform, url: window.location.href, title: document.title }
  }).catch(() => {});

})();
