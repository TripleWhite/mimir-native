// Mimir Background Service Worker
// 处理存储和与 Mimir API 的通信

const MIMIR_API_BASE = 'http://localhost:3000/api'; // Mimir 本地 API 地址
const STORAGE_KEY = 'mimir_conversations';

// 初始化
chrome.runtime.onInstalled.addListener(() => {
  console.log('[Mimir Background] Extension installed');
  
  // 初始化存储
  chrome.storage.local.set({
    [STORAGE_KEY]: [],
    settings: {
      autoCapture: true,
      apiEndpoint: MIMIR_API_BASE,
      syncInterval: 5000
    }
  });
});

// 消息处理中心
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('[Mimir Background] Received message:', request.type, 'from:', sender.tab?.url);
  
  handleMessage(request, sender)
    .then(response => sendResponse(response))
    .catch(error => sendResponse({ success: false, error: error.message }));
  
  return true; // 保持消息通道开放
});

// 消息处理器
async function handleMessage(request, sender) {
  switch (request.type) {
    case 'PAGE_LOADED':
      return await handlePageLoaded(request.data, sender);
      
    case 'CONVERSATION_UPDATED':
      return await handleConversationUpdate(request.data);
      
    case 'GET_CONVERSATIONS':
      return await getStoredConversations();
      
    case 'SEND_TO_MIMIR':
      return await sendToMimirAPI(request.data);
      
    case 'TEST_CONNECTION':
      return await testMimirConnection();
      
    default:
      throw new Error(`Unknown message type: ${request.type}`);
  }
}

// 处理页面加载事件
async function handlePageLoaded(data, sender) {
  console.log(`[Mimir Background] Page loaded: ${data.platform} - ${data.title}`);
  
  // 存储活动会话信息
  await chrome.storage.local.set({
    activeSession: {
      platform: data.platform,
      url: data.url,
      title: data.title,
      tabId: sender.tab?.id,
      loadedAt: new Date().toISOString()
    }
  });
  
  return { success: true, message: 'Page load recorded' };
}

// 处理对话更新
async function handleConversationUpdate(data) {
  try {
    // 先存储到本地
    await storeConversation(data);
    
    // 尝试发送到 Mimir API
    const result = await sendToMimirAPI({
      type: 'conversation',
      data: data
    });
    
    return { success: true, localStored: true, apiResult: result };
  } catch (error) {
    console.error('[Mimir Background] Failed to handle conversation update:', error);
    // 即使 API 失败，本地存储仍然成功
    return { success: true, localStored: true, apiError: error.message };
  }
}

// 存储对话到本地
async function storeConversation(conversation) {
  const { [STORAGE_KEY]: existing = [] } = await chrome.storage.local.get(STORAGE_KEY);
  
  // 查找是否已存在相同 URL 的对话
  const existingIndex = existing.findIndex(c => c.url === conversation.url);
  
  if (existingIndex >= 0) {
    // 更新现有对话
    existing[existingIndex] = {
      ...existing[existingIndex],
      ...conversation,
      updatedAt: new Date().toISOString()
    };
  } else {
    // 添加新对话
    existing.push({
      ...conversation,
      id: generateId(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    });
  }
  
  // 限制存储数量，保留最近 100 条
  const trimmed = existing.slice(-100);
  
  await chrome.storage.local.set({ [STORAGE_KEY]: trimmed });
  console.log('[Mimir Background] Conversation stored, total:', trimmed.length);
  
  return { count: trimmed.length };
}

// 获取存储的对话
async function getStoredConversations() {
  const { [STORAGE_KEY]: conversations = [] } = await chrome.storage.local.get(STORAGE_KEY);
  return { success: true, data: conversations };
}

// 发送到 Mimir API
async function sendToMimirAPI(payload) {
  try {
    const response = await fetch(`${MIMIR_API_BASE}/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log('[Mimir Background] Successfully sent to Mimir API:', result);
    return { success: true, data: result };
  } catch (error) {
    console.error('[Mimir Background] Failed to send to Mimir API:', error);
    throw error;
  }
}

// 测试 Mimir API 连接
async function testMimirConnection() {
  try {
    const response = await fetch(`${MIMIR_API_BASE}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (response.ok) {
      const data = await response.json();
      return { success: true, status: 'connected', data };
    } else {
      return { success: false, status: 'error', message: `HTTP ${response.status}` };
    }
  } catch (error) {
    return { success: false, status: 'disconnected', message: error.message };
  }
}

// 生成唯一 ID
function generateId() {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// 标签页更新监听
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete') {
    const isTargetUrl = tab.url?.includes('claude.ai') || tab.url?.includes('chatgpt.com');
    if (isTargetUrl) {
      console.log('[Mimir Background] Target page loaded:', tab.url);
    }
  }
});

// 定时同步（可选）
chrome.alarms?.onAlarm?.addListener((alarm) => {
  if (alarm.name === 'sync-to-mimir') {
    console.log('[Mimir Background] Scheduled sync triggered');
    // 可以实现定期同步逻辑
  }
});

console.log('[Mimir Background] Service Worker initialized');
