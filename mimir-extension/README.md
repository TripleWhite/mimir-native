# Mimir Chrome Extension

跨 AI 平台记忆助手 - 自动捕获 Claude/ChatGPT 对话

## 功能特性

- 🔗 自动捕获 Claude 和 ChatGPT 对话
- 💾 本地存储 + Mimir API 同步
- 🔍 DOM 变化监听，实时捕获
- 📊 弹出窗口显示统计信息

## 文件结构

```
mimir-extension/
├── manifest.json      # Chrome Extension Manifest V3
├── background.js      # Service Worker - 后台处理
├── content.js         # Content Script - 页面注入
├── inject.js          # 页面主世界注入脚本
├── popup.html         # 弹出窗口 HTML
├── popup.js           # 弹出窗口逻辑
├── README.md          # 说明文档
├── test-extension.js  # 扩展结构验证脚本
└── icons/             # 图标目录
    ├── icon16.png
    ├── icon48.png
    └── icon128.png
```

## 安装步骤

1. 打开 Chrome，进入 `chrome://extensions/`
2. 开启右上角"开发者模式"
3. 点击"加载已解压的扩展程序"
4. 选择 `mimir-extension` 文件夹

## 配置

Mimir API 默认地址：`http://localhost:3000/api`

可以在 background.js 中修改 `MIMIR_API_BASE` 常量。

## 使用方法

1. 安装扩展后，打开 Claude 或 ChatGPT 页面
2. 查看浏览器控制台，应显示 "[Mimir] Extension Loaded on claude" 或 "[Mimir] Extension Loaded on chatgpt"
3. 点击扩展图标，可查看状态、测试连接、手动捕获对话

## 开发

### 消息类型

Content Script -> Background:
- `PAGE_LOADED` - 页面加载完成
- `CONVERSATION_UPDATED` - 对话内容更新

Popup -> Background:
- `TEST_CONNECTION` - 测试 Mimir API 连接
- `GET_CONVERSATIONS` - 获取存储的对话列表
- `SEND_TO_MIMIR` - 手动发送数据到 API

Popup -> Content Script:
- `GET_CONVERSATION` - 获取当前对话
- `PING` - 测试连接

## 验收标准检查清单

- [x] 插件安装成功
- [x] 打开 Claude 页面，console 显示 "[Mimir] Extension Loaded on claude"
- [x] 打开 ChatGPT 页面，console 显示 "[Mimir] Extension Loaded on chatgpt"
- [x] 能发送测试消息到 background

## 验证扩展结构

```bash
cd mimir-extension
node test-extension.js
```