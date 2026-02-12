# 互联网研究汇总 - 记忆系统与 LoCoMo 优化

**研究时间**: 2026-02-12  
**API**: Brave Search  
**目标**: 为 Mimir-Native v2.0 提供最新研究思路

---

## 🔬 LoCoMo 基准测试

### 官方资源
- **论文**: [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753)
- **官网**: https://snap-research.github.io/locomo/
- **GitHub**: https://github.com/snap-research/locomo
- **数据集**: 300 turns, 9K tokens, 35 sessions per conversation

### 关键洞察
1. **生成方法**: 使用 LLM-based agent + persona + temporal event graph
2. **人工验证**: Human annotators verify for long-range consistency
3. **多模态**: Agents can share and react to images
4. **评估任务**:
   - Question Answering
   - Event Summarization
   - Multimodal Dialog Generation

---

## 🧠 记忆系统架构 (MemGPT)

### 核心概念

#### 1. 分层内存系统 (Hierarchical Memory)
```
┌─────────────────────────────────────┐
│      Core Memory (Working Context)   │  ← LLM Context Window
│  - Persona (Agent personality)       │
│  - Human (User information)          │
├─────────────────────────────────────┤
│      Recall Storage                  │  ← Recent messages
├─────────────────────────────────────┤
│      Archival Storage (Vector DB)    │  ← Long-term memory
└─────────────────────────────────────┘
```

#### 2. 虚拟上下文管理 (Virtual Context Management)
- 类比操作系统虚拟内存
- 通过 paging 在不同存储层之间移动数据
- 超出上下文窗口时自动换出

#### 3. 自编辑记忆 (Self-Editing Memory)
- LLM 使用专用工具调用管理记忆
- 可以更新自己的 personality
- 学习用户新信息并更新

#### 4. Heartbeat 机制
- 支持多步推理
- 工具调用后可选 request_heartbeat
- 允许 agent 继续思考

### 记忆管理函数
```python
# Core memory edit
edit_core_memory(section: str, value: str)

# Archival memory operations
insert_archival_memory(content: str)
search_archival_memory(query: str, page: int)

# Recall memory
get_recall_memory(page: int)
```

---

## 🔍 混合检索最佳实践

### RRF (Reciprocal Rank Fusion)

**公式**:
```
RRF_score(d) = Σ(1 / (k + r_i(d)))
```
- k = 60 (常数，防止高排名项过度惩罚)
- r_i(d) = 文档 d 在第 i 个列表中的排名

**加权 RRF**:
```
RRF_score(d) = Σ(w_i * (1 / (k + r_i(d))))
```

### 检索策略组合

| 方法 | 权重 | 用途 |
|------|------|------|
| Vector Search | 40% | 语义相似度 |
| BM25 | 30% | 关键词匹配 |
| Temporal | 20% | 时序相关性 |
| Evidence | 10% | 结构化证据 |

### 优化建议

1. **权重调优**: 根据任务类型动态调整
   - When 问题: 增加 Temporal 权重
   - What 问题: 增加 Vector 权重

2. **Re-Ranking**: 在融合后使用 LLM 重排序
   - 计算与查询的真正相关性
   - 考虑时序上下文

3. **缓存策略**: 
   - 频繁查询结果缓存
   - Embedding 预计算

---

## 🏗️ RAG 长对话处理策略

### 1. 滑动窗口 (Sliding Window)
```python
# 保留最近的 N 轮对话
recent_context = messages[-window_size:]
```

### 2. 摘要压缩 (Summarization)
```python
# 将早期对话压缩为摘要
summary = llm.summarize(old_messages)
context = [summary] + recent_messages
```

### 3. 分层检索 (Hierarchical Retrieval)
```
Level 1: Session-level summaries
Level 2: Turn-level facts
Level 3: Full conversation
```

### 4. 实体链 (Entity Chain)
- 跟踪对话中的关键实体
- 维护实体状态变化
- 用于跨会话引用

---

## 💡 Mimir-Native v2.0 改进建议

### 立即实施

#### 1. 添加 Memory Management Functions
```python
class MimirMemoryManager:
    def edit_core_memory(self, section: str, value: str):
        """编辑核心记忆 (persona/user)"""
        pass
    
    def search_archival(self, query: str, top_k: int = 10):
        """搜索长期记忆"""
        pass
    
    def get_working_context(self) -> List[str]:
        """获取当前工作上下文"""
        pass
```

#### 2. 实现 Dynamic Weight Adjustment
```python
class AdaptiveHybridRetriever:
    def adjust_weights(self, query_intent: QueryIntent):
        """根据查询意图动态调整权重"""
        weights = {
            QueryIntent.WHEN: {'temporal': 0.4, 'evidence': 0.3, 'vector': 0.2, 'bm25': 0.1},
            QueryIntent.WHAT: {'vector': 0.5, 'bm25': 0.3, 'temporal': 0.1, 'evidence': 0.1},
            QueryIntent.WHO: {'bm25': 0.4, 'vector': 0.4, 'temporal': 0.1, 'evidence': 0.1},
        }
        return weights.get(query_intent, self.default_weights)
```

#### 3. 添加 Multi-Step Reasoning
```python
def answer_with_reasoning(query: str, max_steps: int = 3):
    """多步推理回答"""
    for step in range(max_steps):
        # 检索相关信息
        memories = retrieve(query)
        
        # 生成思考
        thought = generate_thought(query, memories)
        
        # 检查是否需要更多信息
        if needs_more_info(thought):
            query = refine_query(query, thought)
            continue
        
        # 生成答案
        return generate_answer(query, memories)
```

### 短期优化

#### 4. 实现 Entity Tracking
```python
class EntityTracker:
    def __init__(self):
        self.entities: Dict[str, EntityState] = {}
    
    def extract_entities(self, text: str) -> List[str]:
        """提取命名实体"""
        pass
    
    def update_entity(self, entity: str, new_state: dict):
        """更新实体状态"""
        pass
    
    def get_entity_history(self, entity: str) -> List[dict]:
        """获取实体历史"""
        pass
```

#### 5. 添加 Session Summarization
```python
class SessionSummarizer:
    def summarize(self, session_data: dict) -> str:
        """生成会话摘要"""
        pass
    
    def incremental_summarize(self, 
                             prev_summary: str, 
                             new_turns: List[dict]) -> str:
        """增量更新摘要"""
        pass
```

#### 6. 实现 Importance Scoring
```python
def calculate_importance(memory: MemoryEntry) -> float:
    """
    计算记忆重要性分数
    因素:
    - 访问频率
    - 最近访问时间
    - 与当前主题的关联度
    - 实体密度
    """
    score = 0.0
    score += min(memory.access_count * 0.1, 0.3)
    score += recency_bonus(memory.last_accessed)
    score += relevance_to_current_topic(memory)
    score += entity_density_score(memory.content)
    return min(score, 1.0)
```

### 中期目标

#### 7. 端到端训练
- 收集 (query, context, answer) 训练数据
- Fine-tune retrieval model
- Train answer generator

#### 8. 多模态支持
- 图像描述索引
- 语音转文本存储
- 视频摘要提取

#### 9. 实时学习
- 从对话中学习用户偏好
- 自动更新 persona
- 错误反馈循环

---

## 📚 参考资源

### 论文
1. [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
2. [LoCoMo: Evaluating Very Long-Term Conversational Memory](https://arxiv.org/abs/2402.17753)
3. [MemoryBench: A Benchmark for Memory and Continual Learning](https://arxiv.org/html/2510.17281v1)

### 框架
- [Letta (MemGPT)](https://www.letta.com/)
- [LlamaIndex Memory](https://www.analyticsvidhya.com/blog/2024/09/memory-and-hybrid-search-in-rag-using-llamaindex/)
- [Haystack Conversational RAG](https://haystack.deepset.ai/cookbook/conversational_rag_using_memory)

### 技术文章
- [Reciprocal Rank Fusion explained](https://medium.com/@devalshah1619/reciprocal-rank-fusion-rrf-explained-in-4-mins-how-to-score-results-form-multiple-retrieval-1a6b2a3b3f2)
- [Elasticsearch RRF](https://www.elastic.co/search-labs/blog/reciprocal-rank-fusion-ranking-problem)

---

## 🎯 下一步行动

1. ✅ **已完成**: Mimir-Native v2.0 基础架构
2. 🔄 **下一步**: 实现 Memory Management Functions
3. 🔄 **下一步**: 添加 Dynamic Weight Adjustment
4. 🔄 **下一步**: 实现 Multi-Step Reasoning
5. 🔄 **下一步**: 运行完整 LoCoMo 测试并分析

---

*研究完成: 2026-02-12*
*Brave API Key 已保存*
