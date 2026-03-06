# KVRM: Registry-Constrained Neural Selection for Verified System Interfaces

This repository contains the code, data, and scripts for reproducing the experiments in the paper:

**"KVRM: Key Value Response Model"**

## Paper

The full paper is available in `paper/kvrm-selection-full.pdf`.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
make reproduce-all

# Run specific experiments
make reproduce-tile-selection    # Tile selection experiments
make reproduce-regret            # Regret analysis
make reproduce-hierarchical      # Hierarchical registry experiments
make reproduce-latency           # Tail latency analysis
```

## Repository Structure

```
kvrm-selection-paper/
├── paper/                   # Paper PDF
├── src/                     # Source code
│   └── kvrm/               # KVRM implementation
│       ├── models/          # KVRM model implementations
│       ├── control_plane/   # Neural control plane components
│       └── utils/           # Utility functions
├── benchmarks/              # Benchmark scripts
├── results/                 # Experimental results
├── scripts/                 # Reproduction scripts
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Experiments

### 1. Tile Selection
GPU kernel tile selection experiments across 4 kernel types (attention, GEMM, LayerNorm, softmax).

**Results:** `results/tile_selection/`

### 2. Regret Analysis
Comprehensive regret analysis comparing KVRM vs heuristic baselines.

**Results:** `results/regret_analysis/`

### 3. Hierarchical Registries
Experiments with hierarchical registries scaling to 10K+ entries.

**Results:** `results/hierarchical/`

### 4. Tail Latency Analysis
P99 latency measurements for real-time viability assessment.

**Results:** `results/latency/`

### 5. ARM64 Instruction Decoding
KVRM applied to ARM64 instruction decoding (95 test cases).

**Results:** `results/arm64/`

## Key Results

- **29% regret reduction** vs static heuristics ($p<0.001$)
- **0% invalid outputs** (architectural guarantee)
- **<$0.3ms selection latency** (P99: 0.026ms)
- **10x-159x improvement** in selection accuracy with hierarchical registries

## Citation

If you use this code or find the paper helpful, please cite:

```bibtex
@article{price2025kvrm,
  title={KVRM: Registry-Constrained Neural Selection for Verified System Interfaces},
  author={Price, Robert},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

Robert Price - bobby@blackweb.ai
