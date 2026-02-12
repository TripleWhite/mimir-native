# OpenClaw 集成指南 - Claude Code Hooks

## 快速开始

### 1. 加载工具
```bash
source /home/ubuntu/.openclaw/workspace/skills/claude-code-hooks/claude-hooks.sh
```

### 2. 运行任务（零轮询）
```bash
# 启动任务（后台运行，立即返回）
claude_hooks_run "实现一个 REST API" ~/myproject

# 显示:
# 🚀 启动 Claude Code (零轮询模式)
#    会话: cc-1234567890-1234
#    任务: 实现一个 REST API...
#    目录: /home/user/myproject
#    PID: 12345
```

### 3. 读取结果
```bash
# 方式 1: 等待完成（低频率轮询）
claude_hooks_wait

# 方式 2: 立即读取最新结果
claude_hooks_read

# 方式 3: 等待特定会话
claude_hooks_wait cc-1234567890-1234
```

## 完整示例

### 场景 1: 简单的后台任务
```bash
# 加载工具
source /home/ubuntu/.openclaw/workspace/skills/claude-code-hooks/claude-hooks.sh

# 启动任务
claude_hooks_run "修复 test_locomo.py 中的日期解析 bug" /tmp/mimir-review/mimir-native

# 任务在后台运行，你可以做其他事情...
# ...

# 稍后检查完成状态
claude_hooks_read
```

### 场景 2: 带 Wake Event 的自动通知
```bash
# 配置 Gateway (用于自动唤醒)
export OPENCLAW_GATEWAY="http://127.0.0.1:18789"
export OPENCLAW_TOKEN="your-gateway-token"

# 启动任务
claude_hooks_run "实现 Hybrid Retriever 优化" ~/workspace

# Claude Code 完成后会自动:
# 1. 写入结果到 ~/.claude-hooks/latest.json
# 2. 发送 Wake Event 到 OpenClaw
# 3. OpenClaw 立即收到通知并处理结果
```

### 场景 3: 并行多任务（Agent Teams）
```bash
source /home/ubuntu/.openclaw/workspace/skills/claude-code-hooks/claude-hooks.sh

# 同时启动多个任务
claude_hooks_run "实现前端界面" ~/project/frontend &
PID1=$!

claude_hooks_run "实现后端 API" ~/project/backend &
PID2=$!

claude_hooks_run "编写测试用例" ~/project/tests &
PID3=$!

echo "三个任务已启动，PID: $PID1, $PID2, $PID3"

# 等待所有完成
wait

# 读取结果
claude_hooks_read
```

## Token 节省对比

| 任务时长 | 传统轮询 | Hooks 方案 | 节省 |
|---------|---------|-----------|------|
| 5 分钟 | ~2,500 tokens | ~100 tokens | 96% |
| 30 分钟 | ~15,000 tokens | ~100 tokens | 99% |
| 2 小时 | ~60,000 tokens | ~100 tokens | 99.8% |

## 文件结构

```
~/.claude-hooks/
├── session-end-hook.sh     # 任务完成时触发
├── stop-hook.sh            # 用户停止时触发
├── latest.json             # 最新执行结果
└── archive/                # 历史结果（可选）
    ├── cc-1234567890-1234.json
    └── cc-1234567891-5678.json
```

## 结果格式 (latest.json)

```json
{
  "session_id": "cc-1739334000-1234",
  "timestamp": "2026-02-12T10:00:00+08:00",
  "cwd": "/home/user/projects/myapp",
  "event": "SessionEnd",
  "status": "done",
  "exit_code": 0,
  "task": "实现用户认证模块",
  "output": "可选: 执行输出摘要"
}
```

## 高级配置

### 自定义输出路径
```bash
export CLAUDE_OUTPUT_FILE="/tmp/my-project-result.json"
claude_hooks_run "任务" ~/myproject
```

### 配置 Gateway 自动唤醒
```bash
# ~/.bashrc
export OPENCLAW_GATEWAY="http://127.0.0.1:18789"
export OPENCLAW_TOKEN="$(cat ~/.openclaw/token)"
```

### 归档历史结果
```bash
# 在 claude-hooks.sh 中添加归档逻辑
archive_result() {
    local archive_dir="$CLAUDE_HOOKS_DIR/archive"
    mkdir -p "$archive_dir"
    cp "$CLAUDE_OUTPUT_FILE" "$archive_dir/cc-$(date +%s).json"
}
```

## 故障排除

### Hook 没有触发
```bash
# 检查环境变量
echo $CLAUDE_SESSION_END_HOOK
echo $CLAUDE_OUTPUT_FILE

# 手动触发 Hook
export CLAUDE_SESSION_ID="test"
export CLAUDE_EXIT_CODE="0"
bash ~/.claude-hooks/session-end-hook.sh
```

### Wake Event 失败
```bash
# 测试 Gateway 连接
curl -X POST "$OPENCLAW_GATEWAY/api/cron/wake" \
  -H "Authorization: Bearer $OPENCLAW_TOKEN" \
  -d '{"text": "测试", "mode": "now"}'
```

### Token 无效
```bash
# 获取 Gateway Token
openclaw gateway token  # 或查看 ~/.openclaw/config.yml
```

## 与现有代码对比

### 之前（轮询方式）
```python
# 问题：频繁轮询消耗大量 Token
process = subprocess.Popen(["claude", "-p", task])
while process.poll() is None:
    time.sleep(5)  # 每 5 秒检查一次
    # 每次检查都消耗上下文 Token
```

### 现在（Hooks 方式）
```bash
# 零轮询，任务完成后自动回调
claude_hooks_run "任务"
# 立即返回，不消耗 Token

# 任务完成后 Hook 自动触发
# OpenClaw 收到 Wake Event 并处理
```

## 下一步

1. ✅ 测试当前实现: `bash test-hooks.sh`
2. ✅ 配置 Gateway Token
3. ✅ 在实际任务中使用
4. 🔄 可选：实现自动归档功能
5. 🔄 可选：添加更多 Hook 类型（Error Hook, Progress Hook）

---

**参考**: [AI超元域博客](https://www.aivi.fyi/aiagents/OpenClaw-Agent-Teams) | [GitHub](https://github.com/win4r/claude-code-hooks)
