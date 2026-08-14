# Benchmark Evaluation Instructions

The following steps describe how to run the benchmark evaluation. Full documentation will be available in a future release.

## Prerequisites

- Git
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Steps

### 1. Clone the repository and install `uv`

```bash
git clone git@github.com:raymondkings/RAM-Newton.git -b evaluation
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
cd RAM-Newton
uv sync --no-cache
```

### 3. Run the benchmark

```bash
uv run python benchmarks/benchmark.py
```

### 4. Collect results

```bash
zip -r evaluation.zip benchmark_results
```

The output archive `evaluation.zip` will contain all benchmark results from the `benchmark_results/` directory.
