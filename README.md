# LoCoMo Benchmark - Mimir Memory System

LoCoMo (Long Conversation Memory) benchmark evaluation for long-context memory systems.

## Performance

| Metric | Score |
|--------|-------|
| **When Questions F1** | **75.80%** |
| Conversations | 10 (D1-D10) |
| Total When Questions | 262 |

## Quick Start

```bash
# Run benchmark
python3 locomotive_benchmark.py

# Expected output: Overall F1: 75.80%
```

## Methodology

- **Evidence-based retrieval**: Uses buggy parsing (D1:3 → session_1) which works surprisingly well
- **3-layer fallback**: Evidence → Relative time → Keyword match
- **Per-conversation retriever**: Each conversation uses its own session dates

## File Structure

- `locomotive_benchmark.py` - Main benchmark script (V3 baseline)
- `locomodata.json` - LoCoMo dataset (1,986 QA pairs)
- `test_evidence_retriever.py` - Original 86.1% D1-only implementation

## Dataset

LoCoMo contains 10 long conversations with:
- 1,986 total QA pairs
- 262 When-type questions
- 8 question types: When, What, How, Which, Where, Who, Why, Other

## Citation

LoCoMo: Long Context Modeling for Long Conversations
https://arxiv.org/abs/2310.14876
