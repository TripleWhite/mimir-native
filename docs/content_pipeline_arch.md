# Mimir 统一内容标注与索引架构

## 核心问题
用户上传的内容多样且混乱：
- AI 对话记录（多轮上下文）
- 社交媒体收藏（短文本/长文/视频链接）
- 本地文件（PDF/图片/语音）
- 实时数据（眼镜拍摄/Plaud录音/微信记录）

**目标**：无论来源/模态，都变成"可调用的记忆"

## 统一处理流水线

```
┌─────────────────────────────────────────────────────────────┐
│                    输入层 (Ingestion)                        │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│ AI 对话      │ 社交媒体     │ 本地文件     │ 实时数据            │
│ (JSON)       │ (URL/Text)  │ (PDF/Img)   │ (Stream)            │
└──────┬──────┴──────┬──────┴──────┬──────┴──────────┬──────────┘
       │             │             │                 │
       └─────────────┴──────┬──────┴─────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 处理层 (Processing)                          │
├─────────────────────────────────────────────────────────────┤
│  1. 模态识别 (Modality Detection)                            │
│     - 文本 / 图像 / 音频 / 视频 / 结构化数据                  │
│                                                              │
│  2. 内容提取 (Content Extraction)                            │
│     - 文本: 直接提取                                         │
│     - 图片: OCR + 视觉描述 (LLM/VLM)                         │
│     - 音频: ASR 转文本                                       │
│     - 视频: 关键帧提取 + ASR                                 │
│     - PDF/文档: 结构化提取 (文本 + 表格 + 图片)               │
│                                                              │
│  3. 语义理解 (Semantic Understanding)                        │
│     - 主题分类 (Topic)                                       │
│     - 实体提取 (Entities: 人/地点/组织/概念)                 │
│     - 情感/意图 (Sentiment/Intent)                           │
│     - 时间信息 (Temporal: 绝对/相对时间)                     │
│     - 重要性评分 (Importance)                                │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 存储层 (Storage)                             │
├─────────────────────────────────────────────────────────────┤
│  记忆单元 (Memory Unit):                                     │
│  {                                                          │
│    id: uuid,                                               │
│    content: "纯文本内容",                                   │
│    content_type: "fact/opinion/plan/preference",           │
│    modality: "text/image/audio/video",                     │
│    source: {                                               │
│      type: "claude/wechat/pdf/plaud/...",                  │
│      url: "...",                                           │
│      timestamp: "..."                                      │
│    },                                                      │
│    entities: ["Caroline", "LGBTQ", "adoption"],            │
│    topics: ["identity", "family", "support"],              │
│    temporal: {                                             │
│      mentioned_date: "2023-05-07",                         │
│      created_at: "2023-05-08T10:00:00Z"                    │
│    },                                                      │
│    embedding: [0.1, 0.2, ...],                             │
│    importance: 0.85,                                       │
│    access_count: 3                                         │
│  }                                                          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 调用层 (Retrieval)                           │
├─────────────────────────────────────────────────────────────┤
│  检索方式:                                                   │
│  - 语义检索: "关于 Caroline 的身份" → 向量相似度搜索         │
│  - 关键词检索: "adoption" → FTS/BM25                       │
│  - 时间检索: "上周的内容" → 时间范围过滤                    │
│  - 实体检索: "涉及 Melanie 的" → 实体标签过滤               │
│  - 混合检索: 语义 + 关键词 + 时间 + 实体                    │
│                                                              │
│  排序策略:                                                   │
│  - 相关性 (向量相似度)                                       │
│  - 时效性 (越新越重要)                                       │
│  - 重要性 (用户标记/系统评分)                                │
│  - 访问频率 (常用内容优先)                                   │
└─────────────────────────────────────────────────────────────┘
```

## 关键设计决策

### 1. 统一表征
所有内容最终都变成：**文本 + 元数据 + 向量**

| 原始内容 | 提取方式 | 存储形式 |
|---------|---------|---------|
| 微信聊天记录 | JSON 解析 | 对话文本 + 时间 + 参与人 |
| 收藏的文章 | 爬虫/Readability | 正文 + 标题 + 摘要 |
| PDF 文件 | pdfplumber/OCR | 结构化文本 + 图片描述 |
| 眼镜视频 | 关键帧 + ASR | 画面描述 + 语音文本 |
| Plaud 录音 | Whisper ASR | 转录文本 + 时间戳 |

### 2. 元数据标准化
无论来源，都有统一的元数据字段：
```python
class MemoryMetadata:
    content_type: Literal["fact", "opinion", "plan", "preference", "event"]
    modality: Literal["text", "image", "audio", "video", "mixed"]
    source: SourceInfo  # 来源平台/文件/设备
    temporal: TemporalInfo  # 时间信息
    entities: List[str]  # 实体标签
    topics: List[str]  # 主题分类
    importance: float  # 0-1
    relations: List[Relation]  # 与其他记忆的关系
```

### 3. 处理触发时机
- **同步处理**: 小文本（对话、短消息）→ 实时处理
- **异步队列**: 大文件（视频、长文档）→ 后台处理
- **增量更新**: 流式数据（微信同步）→ 分批处理

### 4. 调用接口设计
```python
# 统一查询接口
mimir.query(
    query="Caroline 的身份",
    filters={
        "source_type": ["claude", "wechat"],
        "time_range": "last_month",
        "entities": ["Caroline"],
        "topics": ["identity"]
    },
    sort_by="relevance",
    top_k=5
)
```

## 当前 LoCoMo 问题的定位

LoCoMo 表现差不是因为检索，而是因为**处理层**的时序解析失效。

在统一架构下，这属于 "Temporal Info Extraction" 模块：
- 输入："I visited the group yesterday"
- 处理：提取相对时间 + 根据 source.timestamp 计算绝对时间
- 输出：`temporal.mentioned_date = "2023-05-07"`

**修复方向**：
1. 在 Processing 层强制时序解析（不是 LLM 提取后）
2. 每条消息绑定 source.timestamp（session_date）
3. 后处理强制替换（temporal_post_processor）

## 下一步建议

1. **完善 Processing 层**：实现多模态提取管道
2. **修复时序绑定**：确保每条消息都有正确的 source.timestamp
3. **构建 Context Bridge**：让 Claude/Midjourney 能调用这些记忆

你怎么看这个架构？