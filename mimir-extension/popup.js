// Mimir Popup Script
// 处理弹出窗口的交互

document.addEventListener('DOMContentLoaded', async () => {
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  const messageEl = document.getElementById('message');
  const claudeCountEl = document.getElementById('claudeCount');
  const chatgptCountEl = document.getElementById('chatgptCount');

  // 显示消息
  function showMessage(text, type = 'success') {
    messageEl.textContent = text;
    messageEl.className = `message ${type}`;
    setTimeout(() => {
      messageEl.className = 'message';
    }, 3000);
  }

  // 更新状态
  async function updateStatus() {
    try {
      const response = await chrome.runtime.sendMessage({ type: 'TEST_CONNECTION' });
      if (response.success) {
        statusDot.classList.remove('offline');
        statusText.textContent = '已连接到 Mimir';
      } else {
        statusDot.classList.add('offline');
        statusText.textContent = '未连接到 Mimir API';
      }
    } catch (error) {
      statusDot.classList.add('offline');
      statusText.textContent = '扩展未激活';
    }
  }

  // 更新统计
  async function updateStats() {
    try {
      const response = await chrome.runtime.sendMessage({ type: 'GET_CONVERSATIONS' });
      if (response.success && response.data) {
        const claude = response.data.filter(c => c.platform === 'claude').length;
        const chatgpt = response.data.filter(c => c.platform === 'chatgpt').length;
        claudeCountEl.textContent = claude;
        chatgptCountEl.textContent = chatgpt;
      }
    } catch (error) {
      console.error('Failed to get stats:', error);
    }
  }

  // 测试连接按钮
  document.getElementById('testBtn').addEventListener('click', async () => {
    const btn = document.getElementById('testBtn');
    btn.disabled = true;
    btn.textContent = '测试中...';
    
    try {
      const response = await chrome.runtime.sendMessage({ type: 'TEST_CONNECTION' });
      if (response.success) {
        showMessage('连接成功！Mimir API 正常运行', 'success');
      } else {
        showMessage(`连接失败: ${response.message}`, 'error');
      }
    } catch (error) {
      showMessage(`错误: ${error.message}`, 'error');
    }
    
    btn.disabled = false;
    btn.textContent = '测试连接';
    await updateStatus();
  });

  // 手动捕获按钮
  document.getElementById('captureBtn').addEventListener('click', async () => {
    const btn = document.getElementById('captureBtn');
    btn.disabled = true;
    btn.textContent = '捕获中...';
    
    try {
      // 获取当前活动标签页
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      if (!tab.url.includes('claude.ai') && !tab.url.includes('chatgpt.com')) {
        showMessage('请在 Claude 或 ChatGPT 页面使用此功能', 'error');
        return;
      }
      
      // 向 content script 发送消息
      const response = await chrome.tabs.sendMessage(tab.id, { type: 'GET_CONVERSATION' });
      
      if (response.success) {
        // 发送到 background 存储
        await chrome.runtime.sendMessage({
          type: 'SEND_TO_MIMIR',
          data: response.data
        });
        showMessage('对话已捕获并发送到 Mimir', 'success');
        await updateStats();
      } else {
        showMessage('捕获失败', 'error');
      }
    } catch (error) {
      showMessage(`错误: ${error.message}`, 'error');
    }
    
    btn.disabled = false;
    btn.textContent = '手动捕获当前对话';
  });

  // 清除数据按钮
  document.getElementById('clearBtn').addEventListener('click', async () => {
    if (confirm('确定要清除所有本地存储的对话数据吗？')) {
      await chrome.storage.local.set({ mimir_conversations: [] });
      showMessage('本地数据已清除', 'success');
      await updateStats();
    }
  });

  // 初始化
  await updateStatus();
  await updateStats();
});
