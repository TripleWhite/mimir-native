# Mimir-Native

**SQLite-native Memory Layer with Temporal Knowledge Graph**

Mimir-Native is a fully self-contained memory system for AI applications, built entirely on SQLite with vector and full-text search extensions. No external vector databases or graph stores required.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│  Query Interface (Hybrid Retriever)         │
│  - Vector similarity (sqlite-vec)           │
│  - Full-text search (FTS5)                  │
│  - Graph traversal (NetworkX)               │
│  - Temporal filtering                       │
├─────────────────────────────────────────────┤
│  Knowledge Graph (Temporal)                 │
│  - Entity extraction (LLM)                  │
│  - Relation extraction                      │
│  - Temporal resolution                      │
│  - Conflict resolution                      │
├─────────────────────────────────────────────┤
│  Memory Agent                               │
│  - Fact extraction                          │
│  - Deduplication                            │
│  - Embedding generation                     │
├─────────────────────────────────────────────┤
│  Storage (SQLite)                           │
│  - sqlite-vec for vectors                   │
│  - FTS5 for text search                     │
│  - JSON for raw content                     │
└─────────────────────────────────────────────┘
```

## 🚀 Quick Start

```python
from mimir_native import MimirMemory

# Initialize
mimir = MimirMemory(db_path="mimir.db")

# Add content
memories = mimir.add_content(
    content="Caroline visited the LGBTQ support group on May 7, 2023.",
    content_type="conversation"
)

# Search
results = mimir.search(
    query="When did Caroline visit the support group?",
    query_type="temporal"
)
```

## 📦 Installation

```bash
pip install mimir-native
```

## 🧪 LoCoMo Benchmark

Mimir-Native is designed to excel at the LoCoMo benchmark:

```python
from mimir_native.evaluation import LoCoMoEvaluator

evaluator = LoCoMoEvaluator(mimir)
results = evaluator.evaluate("locomo10.json")

print(f"F1 Score: {results['overall']['f1']:.4f}")
print(f"Exact Match: {results['overall']['em']:.4f}")
```

## 🔧 Dependencies

- Python 3.9+
- SQLite 3.35+ (with extension support)
- sqlite-vec
- sentence-transformers (for embeddings)
- networkx (for knowledge graph)

## 📄 License

MIT

## 🔗 Related

- [Mimir Memory](https://github.com/TripleWhite/mimir-memory-v2) - The Mimir ecosystem
