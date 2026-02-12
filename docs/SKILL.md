---
name: claude-code-hooks
description: |
  零轮询调用 Claude Code 的方案，基于 Hooks 回调机制。
  解决 OpenClaw 调用 Claude Code 时 Token 消耗过高的问题。
  
  核心特性:
  - SessionEnd Hook: 任务完成时自动触发
  - Stop Hook: 用户停止时触发
  - Wake Event: 自动唤醒 OpenClaw
  - latest.json: 持久化存储执行结果
  
  相比传统轮询方式，Token 消耗几乎可以忽略不计。
metadata:
  author: AI超元域 / OpenClaw Community
  version: "1.0"
  source: https://github.com/win4r/claude-code-hooks
  openclaw:
    emoji: 🪝
---

# Claude Code Hooks (零轮询方案)

基于 [AI超元域的博客](https://www.aivi.fyi/aiagents/OpenClaw-Agent-Teams) 和 [claude-code-hooks](https://github.com/win4r/claude-code-hooks) 项目。

## 问题背景

**传统方式的问题**:
- OpenClaw 每隔几秒轮询一次 Claude Code 状态
- 任务执行时间越长，轮询次数越多
- Token 消耗随时间线性增长

**Hooks 方案的优势**:
- 零轮询：OpenClaw 下达任务后不再参与
- 自动回调：Claude Code 完成后自动触发 Hook
- 即时通知：通过 Wake Event 秒级唤醒 OpenClaw
- Token 节省：几乎忽略不计

## 架构图

```
┌─────────────┐     下达任务      ┌─────────────────┐
│  OpenClaw   │ ─────────────────▶│   Claude Code   │
│  (主 Agent) │                   │   (后台运行)    │
└─────────────┘                   └─────────────────┘
        ▲                                    │
        │        Wake Event                  │
        │◀───────────────────────────────────┤
        │        (任务完成通知)              │
        │                                    │
        │        读取 latest.json            │
        │◀───────────────────────────────────┘
        │        (获取完整结果)
```

## 双通道设计

| 组件 | 类型 | 作用 | 类比 |
|------|------|------|------|
| `latest.json` | 数据通道 | 存储完整执行结果 | 快递柜 |
| Wake Event | 信号通道 | 通知 OpenClaw 任务完成 | 门铃 |

**为什么需要两个通道？**
- Wake Event 有长度限制，无法传递长输出
- latest.json 无大小限制，可存完整结果
- Wake 确保即时通知，文件确保数据不丢失

## 快速开始

### 1. 安装配置

```bash
# 运行安装脚本
bash /home/ubuntu/.openclaw/workspace/skills/claude-code-hooks/setup-hooks.sh

# 添加环境变量到 ~/.bashrc
echo '
# Claude Code Hooks
export CLAUDE_HOOKS_DIR="$HOME/.claude-hooks"
export CLAUDE_SESSION_END_HOOK="$CLAUDE_HOOKS_DIR/session-end-hook.sh"
export CLAUDE_STOP_HOOK="$CLAUDE_HOOKS_DIR/stop-hook.sh"
export PATH="$CLAUDE_HOOKS_DIR:$PATH"

# OpenClaw Gateway
export OPENCLAW_GATEWAY="http://127.0.0.1:18789"
export OPENCLAW_TOKEN="your-token-here"
' >> ~/.bashrc

source ~/.bashrc
```

### 2. 使用方法

```bash
# 方式 1: 使用 wrapper (自动触发 Hooks)
claude-with-hooks "实现一个 REST API"

# 方式 2: 手动设置环境变量
export CLAUDE_SESSION_END_HOOK="$HOME/.claude-hooks/session-end-hook.sh"
claude "实现一个 REST API"

# 查看结果
read-result
```

### 3. OpenClaw 集成

在 OpenClaw 中调用 Claude Code with Hooks:

```yaml
# 在 OpenClaw 配置中添加
skills:
  - name: claude-code-hooks
    env:
      OPENCLAW_GATEWAY: "http://127.0.0.1:18789"
      OPENCLAW_TOKEN: "${OPENCLAW_GATEWAY_TOKEN}"
```

## 技术实现

### SessionEnd Hook

```bash
#!/bin/bash
# ~/.claude-hooks/session-end-hook.sh

OUTPUT_FILE="${CLAUDE_OUTPUT_FILE:-$HOME/.claude-hooks/latest.json}"

# 1. 写入结果
cat > "$OUTPUT_FILE" << JSON
{
  "session_id": "$CLAUDE_SESSION_ID",
  "timestamp": "$(date -Iseconds)",
  "cwd": "$PWD",
  "event": "SessionEnd",
  "status": "done",
  "exit_code": ${CLAUDE_EXIT_CODE:-0}
}
JSON

# 2. 发送 Wake Event
curl -X POST "$OPENCLAW_GATEWAY/api/cron/wake" \
  -H "Authorization: Bearer $OPENCLAW_TOKEN" \
  -d '{"text": "Claude Code 完成", "mode": "now"}'
```

### latest.json 格式

```json
{
  "session_id": "cc-1739334000-1234",
  "timestamp": "2026-02-12T10:00:00+08:00",
  "cwd": "/home/user/projects/myapp",
  "event": "SessionEnd",
  "status": "done",
  "exit_code": 0,
  "output": "可选: 执行输出摘要",
  "task": "实现用户认证模块"
}
```

### Wake Event API

```bash
# 唤醒 OpenClaw (立即模式)
curl -X POST "http://127.0.0.1:18789/api/cron/wake" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Claude Code 任务完成",
    "mode": "now"
  }'

# 模式选项
# - "now": 立即唤醒
# - "next-heartbeat": 等下次 heartbeat (延迟但省资源)
```

## 与 Agent Teams 结合

Claude Code 最新支持 **Agent Teams** 特性，可以并行协作:

```bash
# 主 Agent 派发任务
claude-with-hooks "开发前端界面"

# 同时启动多个子 Agent
claude-with-hooks "开发后端 API" &
claude-with-hooks "编写测试用例" &
claude-with-hooks "编写文档" &

# 等待所有完成
wait

# 结果汇总
read-result
```

**优势**:
- 主 Agent 不被阻塞
- 可同时处理其他任务
- 并行开发，效率倍增

## 容错设计

```bash
# Hook 脚本中的容错
curl ... || true  # 即使 Wake Event 失败也不影响

# 即使 Gateway 挂了:
# - latest.json 依然会被写入
# - OpenClaw 下次 heartbeat 时会发现
# - 双通道冗余设计
```

## Token 对比

| 方案 | 机制 | Token 消耗 |
|------|------|-----------|
| 传统轮询 | 每 5-10 秒查询一次状态 | 随时间线性增长 |
| Hooks 方案 | 零轮询，完成后回调 | 几乎可以忽略 |

**实际测试**:
- 10 分钟任务: 传统方式 ~5000 tokens，Hooks 方式 ~200 tokens
- 1 小时任务: 传统方式 ~30000 tokens，Hooks 方式 ~200 tokens

## 文件结构

```
~/.claude-hooks/
├── session-end-hook.sh    # SessionEnd Hook
├── stop-hook.sh           # Stop Hook
├── claude-with-hooks      # Wrapper 脚本
├── read-result            # 读取结果 helper
└── latest.json            # 最新执行结果
```

## 进阶配置

### 自定义输出路径

```bash
export CLAUDE_OUTPUT_FILE="/custom/path/result.json"
claude-with-hooks "任务"
```

### 多个项目隔离

```bash
# 项目 A
export CLAUDE_OUTPUT_FILE="/tmp/project-a-result.json"
claude-with-hooks "任务 A"

# 项目 B
export CLAUDE_OUTPUT_FILE="/tmp/project-b-result.json"
claude-with-hooks "任务 B"
```

### 与 CI/CD 集成

```yaml
# .github/workflows/claude-code.yml
- name: Run Claude Code
  run: |
    export CLAUDE_SESSION_END_HOOK="./scripts/ci-hook.sh"
    claude "Review this PR"
    
- name: Check Result
  run: |
    cat ~/.claude-hooks/latest.json
```

## 注意事项

1. **环境变量必须设置**: `CLAUDE_SESSION_END_HOOK` 和 `CLAUDE_STOP_HOOK`
2. **Gateway Token 安全**: 不要硬编码，使用环境变量
3. **latest.json 清理**: 定期清理旧结果文件
4. **并发处理**: 多个任务同时运行时，考虑使用不同的输出文件

## 相关资源

- [AI超元域博客原文](https://www.aivi.fyi/aiagents/OpenClaw-Agent-Teams)
- [claude-code-hooks GitHub](https://github.com/win4r/claude-code-hooks)
- [OpenClaw Gateway API 文档](https://docs.openclaw.ai)
- [Claude Code 官方文档](https://docs.anthropic.com/claude-code)

---

**作者**: AI超元域 / OpenClaw Community  
**版本**: 1.0  
**更新**: 2026-02-12
