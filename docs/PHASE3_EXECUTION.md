# Mimir Phase 3: Chrome Extension - 自动记忆助手

**最终方案确认**: 自动捕获 + 自然语言取用（不替代主界面）

---

## 🎯 核心交互

```
用户使用 Claude/ChatGPT（正常界面）
         ↓
Mimir 后台自动捕获（静默，无感知）
         ↓
需要历史时：
  ├─ 侧边栏: "@Mimir 找上周的方案" → 选择插入
  └─ 快捷指令: "/mimir 找架构讨论" → 自动插入
```

---

## 📅 30 天执行计划

### Week 1: 自动捕获基础 (02-11 ~ 02-17)

**Day 1 (明天)**: Chrome Extension 基础架构
- [ ] manifest.json v3 配置
- [ ] content script 注入 Claude/ChatGPT 页面
- [ ] background service worker
- [ ] 与 Mimir API 通信

**Day 2-3**: Claude 自动捕获
- [ ] 监听对话 DOM 变化
- [ ] 提取消息内容、时间、角色
- [ ] 实时发送到 Mimir
- [ ] 测试：对话自动存入

**Day 4-5**: ChatGPT 自动捕获
- [ ] 适配 ChatGPT 页面结构
- [ ] 同样逻辑捕获对话
- [ ] 测试：双平台同时捕获

**Day 6-7**: 捕获优化
- [ ] 处理长对话分页
- [ ] 增量捕获（只抓新消息）
- [ ] 错误处理与重试
- [ ] Week 1 验收测试

**Week 1 交付**:
- Chrome Extension 自动捕获 Claude + ChatGPT
- 对话实时同步到 Mimir
- 无感知后台运行

---

### Week 2: 侧边栏与自然语言 (02-18 ~ 02-24)

**Day 8-10**: 侧边栏界面
- [ ] 侧边栏 UI 框架
- [ ] @Mimir 聊天界面
- [ ] 自然语言输入
- [ ] 检索结果显示

**Day 11-12**: 一键插入
- [ ] 选择记忆项
- [ ] 格式化插入当前页面输入框
- [ ] 摘要 vs 原文选项

**Day 13-14**: 侧边栏测试
- [ ] Claude 页面侧边栏
- [ ] ChatGPT 页面侧边栏
- [ ] 跨平台记忆调取
- [ ] Week 2 验收

---

### Week 3: 快捷指令 (02-25 ~ 03-03)

**Day 15-17**: /mimir 指令
- [ ] 监听输入框 "/mimir"
- [ ] 指令解析与检索
- [ ] 下拉提示与选择
- [ ] 自动插入结果

**Day 18-19**: 快捷指令优化
- [ ] 模糊匹配
- [ ] 历史指令记忆
- [ ] 智能补全

**Day 20-21**: 双模式整合
- [ ] 侧边栏 + 快捷指令并用
- [ ] 用户偏好设置
- [ ] Week 3 验收

---

### Week 4: 智能主动 (03-04 ~ 03-10)

**Day 22-25**: 主动提示
- [ ] 检测上下文缺失
- [ ] 智能匹配相关记忆
- [ ] 非侵入式提示气泡
- [ ] 用户确认后插入

**Day 26-28**: 学习优化
- [ ] 分析用户取用习惯
- [ ] 优化检索排序
- [ ] 个性化推荐

**Day 29-30**: 演示与文档
- [ ] 演示视频录制
- [ ] 技术博客
- [ ] Chrome Web Store 准备

---

## 🚀 明天开始 Day 1

**任务**: Chrome Extension 基础架构

**具体工作**:
1. 创建 `mimir-extension/` 目录
2. 配置 manifest.json v3
3. 创建 content script 注入逻辑
4. 搭建 background service worker
5. 测试：插件能注入 Claude 和 ChatGPT 页面

**验收标准**:
- [ ] 插件安装成功
- [ ] 打开 Claude 页面，console 能看到 "Mimir Extension Loaded"
- [ ] 打开 ChatGPT 页面，同样显示
- [ ] 能发送测试消息到 background

**开始开发？**