# Zym

Experimenting with code search using CUDA for acceleration. Currently supports exact and fuzzy pattern matching.

## Building

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### CLI Tool (zym)

```bash
# Basic search
./zym /path/to/code TODO

# GPU search
./zym --gpu /path/to/code main

# Fuzzy search (max edit distance 2)
./zym --fuzzy 2 /path/to/code FIXME

# Verbose output with timing
./zym --gpu --verbose --time /path/to/code TODO
```

**Options:**
- `--cpu` - Use CPU (default)
- `--gpu` - Use GPU
- `--fuzzy <N>` - Fuzzy matching with max distance N
- `--threads <N>` - Number of CPU threads
- `--verbose` - Show execution details
- `--time` - Show timing info
- `--help` - Help message

> fuzzy matching on the CPU is very slow. Prefer the GPU implementation.

### Benchmarking

Create a config file (`benchmark.conf`):
```
# name,path,patterns,exact|fuzzy,max_distance
small,/path/to/code,TODO:FIXME:int,exact,0
fuzzy_test,/path/to/code,TODO:main,fuzzy,1
```

Run benchmarks:
```bash
./benchmark --config benchmark.conf --output results.csv --runs 10
```

**Options:**
- `--config <file>` - Config file (default: benchmark.conf)
- `--output <file>` - Output CSV (default: benchmark.csv)
- `--runs <N>` - Iterations per test (default: 5)
- `--include-fuzzy-cpu` - Include fuzzy CPU tests (because it's excluded by default as it's slow)