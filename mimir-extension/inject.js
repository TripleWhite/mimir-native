// Mimir Inject Script - 注入页面主世界
// 用于访问页面内部的 JavaScript 变量

(function() {
  'use strict';

  console.log('[Mimir Inject] Script loaded into page context');

  // 检测平台
  const platform = window.location.hostname.includes('claude.ai') ? 'claude' : 'chatgpt';

  // 向 content script 发送消息
  function notifyContentScript(type, data) {
    window.postMessage({
      type: `MIMIR_${type}`,
      data: data,
      source: 'mimir-inject',
      timestamp: Date.now()
    }, '*');
  }

  // 监听页面路由变化
  let lastUrl = location.href;
  new MutationObserver(() => {
    const url = location.href;
    if (url !== lastUrl) {
      lastUrl = url;
      console.log('[Mimir Inject] URL changed:', url);
      notifyContentScript('URL_CHANGED', { url, platform });
    }
  }).observe(document, { subtree: true, childList: true });

  // Claude 特定的监听
  if (platform === 'claude') {
    // 监听 Claude 的流式响应完成
    const originalFetch = window.fetch;
    window.fetch = async function(...args) {
      const response = await originalFetch.apply(this, args);
      
      // 检测对话相关的 API 调用
      const url = args[0];
      if (typeof url === 'string' && url.includes('api')) {
        console.log('[Mimir Inject] Claude API call detected:', url);
        notifyContentScript('API_CALL', { url: url.toString(), method: args[1]?.method || 'GET' });
      }
      
      return response;
    };
  }

  // ChatGPT 特定的监听
  if (platform === 'chatgpt') {
    // 监听 ChatGPT 的 SSE 流
    const originalEventSource = window.EventSource;
    window.EventSource = function(...args) {
      const es = new originalEventSource(...args);
      console.log('[Mimir Inject] ChatGPT EventSource created:', args[0]);
      notifyContentScript('EVENTSOURCE_CREATED', { url: args[0] });
      return es;
    };
    window.EventSource.prototype = originalEventSource.prototype;
  }

  // 导出全局变量供调试
  window.__MIMIR_DEBUG = {
    version: '1.0.0',
    platform: platform,
    loaded: true,
    timestamp: new Date().toISOString()
  };

  console.log('[Mimir Inject] Setup complete for', platform);
})();
