# Supermemory SOTA 架构深度解析

**来源**: https://supermemory.ai/research  
**基准**: LongMemEval_s (比 LoCoMo 更严格)  
**性能**: Multi Session 71.43%, Temporal Reasoning 76.69%

---

## 🏆 核心创新

### 1. Chunk-based Ingestion + Contextual Memories

**问题**: 标准 RAG 检索原始 chunks 缺乏上下文

**解决方案**:
```
原始对话 → Chunking → Memory Generation → 存储
                ↓
         Contextual Retrieval (Anthropic)
         
Memory = Atomic piece of information
       + 解析模糊引用
       + 上下文信息
```

**Mimir-Native 应用**:
```python
class ContextualMemoryGenerator:
    def generate_memory(self, chunk: str, context: str) -> Memory:
        """
        生成带上下文的记忆
        不是简单存储 chunk，而是提取 atomic fact
        """
        atomic_fact = self.llm.extract_fact(chunk, context)
        return Memory(
            content=atomic_fact,
            source_chunk=chunk,
            context_summary=context[:200]
        )
```

---

### 2. Relational Versioning (知识版本控制)

**三种关系类型**:

| 关系 | 描述 | 示例 |
|------|------|------|
| **updates** | 状态突变，处理矛盾 | "favorite color is now Green" → 更新 Blue |
| **extends** | 补充细节，无矛盾 | 添加 job title 到 employment memory |
| **derives** | 二阶逻辑推断 | 从多个 memories 推断新信息 |

**Mimir-Native 应用**:
```python
class RelationalMemoryGraph:
    def add_relation(self, new_memory: Memory, existing: Memory):
        if self.is_contradiction(new_memory, existing):
            relation = RelationType.UPDATES
            self.version_history[existing.id].append(new_memory)
        elif self.is_supplement(new_memory, existing):
            relation = RelationType.EXTENDS
        elif self.is_inference(new_memory, [existing]):
            relation = RelationType.DERIVES
            
        self.graph.add_edge(existing, new_memory, relation)
```

---

### 3. Temporal Grounding (双重时间戳)

**关键洞察**: 每个记忆有两个时间戳

```python
@dataclass
class TemporalMetadata:
    documentDate: datetime  # 对话发生时间
    eventDate: List[datetime]  # 事件实际发生时间
```

**用途**:
- **documentDate**: 计算相对时间 ("yesterday" relative to documentDate)
- **eventDate**: 真实事件时序

**Mimir-Native 应用**:
```python
class TemporalGrounding:
    def parse_relative_time(self, text: str, document_date: datetime) -> datetime:
        """
        "yesterday" → document_date - 1 day
        NOT current date!
        """
        pass
    
    def extract_event_dates(self, text: str) -> List[datetime]:
        """提取文本中所有事件时间"""
        pass
```

---

### 4. Hybrid Search Strategy (混合搜索)

**两阶段搜索**:

```
阶段 1: Semantic Search on Memories
        ↓
   找到相关 memory (高 signal, 低 noise)
   
阶段 2: Inject Source Chunk
        ↓
   返回原始 chunk 给 LLM (finer details)
```

**优势**:
- Memories 是 atomic facts → 高精确度检索
- Chunks 提供完整上下文 → LLM 有足够细节

**Mimir-Native 应用**:
```python
class TwoStageRetriever:
    def retrieve(self, query: str, top_k: int = 10):
        # Stage 1: Search memories
        memories = self.memory_store.search(query, top_k=top_k*2)
        
        # Stage 2: Get source chunks
        results = []
        for mem in memories:
            chunk = self.chunk_store.get(mem.source_chunk_id)
            results.append({
                'memory': mem.content,
                'chunk': chunk.content,
                'temporal': chunk.temporal_metadata
            })
        
        return results
```

---

## 📊 LongMemEval 基准

### 为什么比 LoCoMo 更严格？

| 特性 | LoCoMo | LongMemEval |
|------|--------|-------------|
| 上下文长度 | 有限 | 115k+ tokens |
| 知识更新 | ❌ | ✅ (overwrite old info) |
| 人类-助手对话 | ❌ | ✅ (更像真实使用) |
| 噪声环境 | 低 | 高 |

### 评估类别

1. **single-session-user**: 检索用户提到的内容
2. **single-session-assistant**: 检索助手提到的内容
3. **single-session-preference**: 提取用户偏好
4. **multi-session**: 跨会话推理
5. **knowledge-update**: 知识更新处理
6. **temporal-reasoning**: 时序推理

---

## 🛠️ 实现路线图

### Phase 1: Contextual Memories (立即)

```python
# 修改现有的 fact extraction
class ContextualFactExtractor:
    def extract(self, text: str, context: str) -> List[Memory]:
        prompt = f"""
        Extract atomic facts from this text.
        Resolve any ambiguous references using the context.
        
        Text: {text}
        Context: {context}
        
        Output format:
        - Fact: [clear, standalone fact]
        - Source: [original text span]
        """
        return self.llm.extract(prompt)
```

### Phase 2: Relational Graph (短期)

```python
# 添加关系跟踪
class MemoryGraph:
    def __init__(self):
        self.nodes: Dict[str, Memory] = {}
        self.edges: List[Relation] = []
        self.versions: Dict[str, List[Memory]] = {}
    
    def add_memory(self, memory: Memory):
        # Check for relations with existing memories
        for existing in self.nodes.values():
            relation = self.detect_relation(memory, existing)
            if relation:
                self.edges.append(Relation(existing, memory, relation))
                
        self.nodes[memory.id] = memory
```

### Phase 3: Temporal Grounding (短期)

```python
# 增强 temporal_normalizer.py
class TemporalGrounding:
    def __init__(self):
        self.document_date: Optional[datetime] = None
        self.event_dates: List[datetime] = []
    
    def set_document_date(self, date: datetime):
        """设置文档日期作为相对时间基准"""
        self.document_date = date
    
    def parse_relative(self, text: str) -> datetime:
        """相对于 document_date 解析"""
        if "yesterday" in text.lower():
            return self.document_date - timedelta(days=1)
        # ... more patterns
```

### Phase 4: Two-Stage Retrieval (中期)

```python
# 实现两阶段搜索
class SupermemoryRetriever:
    def __init__(self):
        self.memory_index = MemoryIndex()  # 轻量级，高 signal
        self.chunk_index = ChunkIndex()    # 完整 chunks
    
    def search(self, query: str):
        # Stage 1: Fast memory search
        memories = self.memory_index.search(query, top_k=20)
        
        # Stage 2: Fetch chunks
        results = []
        for mem in memories:
            chunk = self.chunk_index.get(mem.chunk_id)
            results.append({
                'fact': mem.content,
                'details': chunk.content,
                'when': chunk.temporal
            })
        
        return results
```

---

## 💡 关键洞察

### 1. 最小化语义歧义
> "Supermemory achieves SOTA by minimizing semantic ambiguity"

**方法**: 将 memories 与时间元数据、关系、原始 chunks 耦合

### 2. Session-Based Ingestion
> "We ingest session-by-session, not round-by-round"

**优势**: 保留会话级别的上下文和连贯性

### 3. Knowledge Chains
通过关系链接形成知识演化历史

---

## 📚 参考

- **Supermemory Research**: https://supermemory.ai/research
- **LongMemEval Paper**: Wu et al., 2024
- **Anthropic Contextual Retrieval**: https://www.anthropic.com/engineering/contextual-retrieval
- **Zep Memory**: Rasmussen et al., 2025

---

## 🎯 应用到 Mimir-Native

### 立即可以做的改进

1. ✅ **已有**: Evidence-based retrieval (86.1% F1)
2. 🔄 **添加**: Contextual memory generation
3. 🔄 **添加**: Dual-layer timestamp (documentDate + eventDate)
4. 🔄 **添加**: Relation tracking (updates/extends/derives)
5. 🔄 **添加**: Two-stage retrieval (memory → chunk)

### 预期提升

| 改进 | 当前 | 预期 |
|------|------|------|
| Contextual Memories | 86.1% | 88%+ |
| Temporal Grounding | 86.1% | 89%+ |
| Relational Graph | 86.1% | 90%+ |

---

*分析完成: 2026-02-12*
*下一步: 实现 Phase 1-2*
