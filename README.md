# KVRM: Key-Value Response Mapping

### Neural networks that *cannot* produce invalid outputs.

---

## The Problem

Every system that uses AI to make decisions faces the same trade-off:

| Approach | Understands Language | Bounded Output | The Catch |
|----------|---------------------|----------------|-----------|
| **Regex / Parsers** | No | Yes | Breaks on typos, paraphrases, informal input |
| **LLMs (GPT-4, etc.)** | Yes | No | Can hallucinate dangerous outputs like `DROP TABLE users;` |

Deterministic parsers are perfectly reliable but shatter on natural language. LLMs understand everything but can say *anything* — including dangerous, invalid, or nonsensical outputs. There is no middle ground.

**KVRM is the middle ground.**

## What KVRM Does

KVRM (Key-Value Response Mapping) trains neural networks to **classify inputs into a pre-approved vocabulary** instead of generating free-form text. The output is always a valid key from a fixed registry — by construction, not by post-hoc validation.

```
User says: "show me all users"           --> KVRM outputs: SELECT_ALL_USERS  (valid key)
User says: "can I see the user list?"    --> KVRM outputs: SELECT_ALL_USERS  (valid key)
User says: "retreive users" (typo)       --> KVRM outputs: SELECT_ALL_USERS  (valid key)
User says: "asdkjh garbage input"        --> KVRM outputs: INVALID           (valid key)
                                             ↑ Can NEVER output "DROP TABLE users"
```

The key insight: **KVRM transforms unbounded generation risk into bounded classification risk.** The worst thing that can happen is a *misclassification within your approved vocabulary* — never an arbitrary hallucination.

## Results

**Primary validation**: 10 semantic tasks, 10,400 examples, 5-fold cross-validation, 5 seeds (250 measurements):

| Metric | Result |
|--------|--------|
| **Overall accuracy** | **99.1%** (structured: 100%, semantic: 100%, edge cases: 90%) |
| **Invalid outputs** | **0%** — architecturally impossible |
| **Latency** | **0.04ms** per classification |
| **Parameters** | **~2K** (vs millions for LLMs) |
| **Statistical power** | **0.95** (exceeds 0.80 standard) |

Compared to alternatives:

| Approach | Accuracy | Handles Typos/Paraphrases | Can Output Invalid SQL | Latency |
|----------|----------|--------------------------|------------------------|---------|
| Regex | 100% | No | No | 0.001ms |
| **KVRM** | **99.1%** | **Yes** | **No** | **0.04ms** |
| GPT-4 | ~95-98% | Yes | **Yes** | ~200ms |
| Constrained decoding (Outlines) | ~97% | Yes | No | ~110ms |

## Why This Matters

**For safety-critical systems**: Medical interfaces, financial systems, industrial control — anywhere an invalid output could cause harm. KVRM guarantees outputs stay within your approved action space.

**For production NLP**: Replace brittle regex pipelines with semantic understanding while keeping bounded outputs. KVRM handles typos, paraphrasing, slang, and informal language that breaks deterministic parsers.

**For edge deployment**: At ~2K parameters and 0.04ms latency, KVRM runs anywhere — phones, microcontrollers, embedded systems. No GPU required.

**For cost reduction**: The paper documents **52-95% cost reduction** vs LLM-based alternatives across 6 industry use cases (healthcare, finance, smart home, customer support, industrial IoT, database admin).

## How It Works (30 Seconds)

1. **Define a vocabulary registry** — the set of valid actions your system can take (e.g., `{SELECT_USERS, DELETE_USER, CREATE_USER, INVALID}`)
2. **Train a small classifier** — any architecture (SVM, neural net, fine-tuned LLM) that maps natural language to registry keys
3. **Wire keys to verified functions** — each key maps to a pre-audited, tested function with known behavior

The output layer has exactly N neurons (one per registry key) followed by softmax. It is *mathematically impossible* to produce an output outside the registry.

```
Input: "show me all users"
  → Encoder → [0.01, 0.97, 0.01, 0.01]  (4 keys)
  → argmax → key: SELECT_ALL_USERS
  → Registry → executes pre-verified SELECT query
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
make reproduce-all

# Run specific experiments
make reproduce-tile-selection    # GPU tile selection experiments
make reproduce-regret            # Regret analysis vs heuristic baselines
make reproduce-hierarchical      # Hierarchical registry (10K+ entries)
make reproduce-latency           # Tail latency (P99: 0.026ms)
```

## Repository Structure

```
kvrm-selection-paper/
├── paper/                   # Full paper PDF
├── src/kvrm/                # KVRM implementation
│   ├── models/              # KVRM model implementations
│   └── control_plane/       # Neural control plane components
├── results/                 # Pre-computed experimental results
│   ├── tile_selection/      # GPU kernel tile selection
│   ├── regret_analysis/     # KVRM vs heuristic baselines
│   ├── hierarchical/        # Scaling to 10K+ registry entries
│   ├── latency/             # P99 tail latency measurements
│   └── arm64/               # ARM64 instruction decoding
├── requirements.txt
└── Makefile                 # One-command experiment reproduction
```

## Paper

The full paper is available at [`paper/kvrm-selection-full.pdf`](paper/kvrm-selection-full.pdf).

**Secondary validation (stress test)**: As a precision ceiling test, KVRM was used to build a 64-bit ARM64 neural CPU — 8 specialist networks (234K parameters total) that decode and execute real CPU instructions with **100% accuracy** on 4,200 tests, including bubble sort, binary search, encryption, and DOOM-style ray casting. Every output is a valid instruction — always.

## When to Use KVRM

| Scenario | Use KVRM? | Why |
|----------|-----------|-----|
| Semantic input + bounded output needed | **Yes** | Core use case |
| Edge deployment, low latency required | **Yes** | 0.04ms, ~2K params |
| Safety-critical system | **Yes** | 0% invalid outputs by construction |
| Perfectly structured input (no variation) | No | Regex is simpler and 100% accurate |
| Open-ended text generation | No | Use an LLM |
| High adversarial threat model | No | 79.5% adversarial robustness is insufficient |

## Citation

```bibtex
@article{price2026kvrm,
  title={KVRM: Key-Value Response Mapping — Bounded Neural Outputs for Safety-Critical Semantic Classification},
  author={Price, Robert},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

Robert Price — bobby@blackweb.ai
