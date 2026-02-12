# LoCoMo Benchmark - Mimir Memory System

Long Conversation Memory benchmark evaluation with LLM enhancement.

## Performance (V5 Final)

| Metric | Score |
|--------|-------|
| **Overall F1** | **72.2%** |
| When | 79.8% |
| Why | 89.8% |
| What | 74.2% |
| Who | 68.6% |
| Where | 66.5% |
| How | 64.2% |
| Which | 53.8% |
| **Other** | **61.3%** |

## Quick Start

```bash
python3 locomotive_benchmark_final.py
# Expected: Overall F1: 72.15%
```

## Methodology

- **When/Why/What/Who/Where**: V3 evidence-based retrieval
- **How/Other/Which**: LLM enhancement with context extraction
- **346 LLM calls** for 346 Which/Other/How questions

## Key Improvements vs V3

| Type | V3 | V5 | Delta |
|------|-----|-----|-------|
| Other | 38.5% | 61.3% | +22.8% |
| How | 59.5% | 64.2% | +4.7% |
| Which | 54.5% | 53.8% | -0.7% |
| **Overall** | 68.2% | **72.2%** | **+4.0%** |

## Files

- `locomotive_benchmark_final.py` - V5 final (72.2% F1)
- `locomotive_benchmark.py` - V3 baseline (75.8% When-only)
- `locomodata.json` - LoCoMo dataset

## Dataset

- 10 conversations (D1-D10)
- 1,986 QA pairs
- 262 When-type questions
- 8 question types total

## Citation

LoCoMo: Long Context Modeling for Long Conversations
https://arxiv.org/abs/2310.14876
