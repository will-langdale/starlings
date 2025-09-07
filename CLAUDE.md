# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

starlings is a hybrid Python/Rust package for systematically exploring and comparing entity resolution results across different thresholds and methods. Instead of forcing threshold decisions at processing time, Starlings preserves the complete resolution space as a hierarchy of merge events, enabling instant threshold exploration, efficient metric computation, and lossless data transport between pipeline stages.

### Core Innovation

Starlings revolutionises entity resolution by storing **merge events** rather than fixed clusters, enabling instant exploration of any threshold without recomputation. This achieves 10-100x performance improvements through O(k) incremental metric updates.

### Key Technical Architecture

**Multi-Collection Model**:
- EntityFrame = (Records, {Hierarchies}, Interning)
- Collections ARE hierarchies that generate partitions at any threshold
- Contextual ownership enables memory sharing between collections

**Performance Characteristics**:
- Hierarchy construction: O(m log m) where m = edges
- Threshold query: O(m) first time, O(1) cached
- Metric updates: O(k) incremental between thresholds
- Memory: ~60-115MB for 1M edges
- **Production performance**: 1M edges processed in <1.6s (PGO build), 645k edges/second throughput

**Implementation Stack**:
- **Rust core**: Performance-critical operations using lock-free data structures (boxcar::Vec), parking_lot, FxHasher, Rayon parallelisation
- **Python interface**: Polars-inspired wrapper pattern via PyO3
- **Arrow integration**: Efficient serialisation with dictionary encoding
- **Key optimisations**: Lock-free DataContext, cached hashing, parallel key interning, PGO support

## Architecture

The project uses a professional two-crate architecture following the Polars pattern:

```
src/
├── python/starlings/       # Python package with main API
├── rust/
│   ├── starlings-core/     # Pure Rust crate (zero PyO3 dependencies)
│   │   ├── src/core/       # Data structures, algorithms
│   │   ├── src/hierarchy/  # Partition hierarchies, merge events
│   │   └── benches/        # Performance benchmarks
│   └── starlings-py/       # Minimal PyO3 wrapper
│       └── src/lib.rs      # Python bindings only
└── tests/                  # Python integration test suite
```

**Build system**: 
- Root `Cargo.toml`: Workspace configuration for both Rust crates
- Root `pyproject.toml`: Python package config, maturin points to starlings-py
- `starlings-core`: Pure Rust business logic, can be used by other Rust projects
- `starlings-py`: Thin PyO3 wrapper over starlings-core, exports PyCollection and PyPartition classes

## Development commands

This project uses `just` as a command runner with a modular structure and `uv` for Python dependency management:

### Core commands
- `just install`: Install development dependencies
- `just format`: Format and lint all code (Python + Rust)
- `just clean`: Clean build artifacts
- `just docs`: Run a local documentation development server

### Modular command structure
- `just build`: Build the Rust extension (see `just build list` for options)
  - `just build`: Standard development build
  - `just build pgo`: Profile-Guided Optimisation build for maximum performance
- `just test`: Run test suite (see `just test list` for options)
  - `just test`: Run all tests (Python integration + Rust core)
  - `just test python`: Run Python integration tests only
  - `just test rust`: Run Rust core tests only
- `just bench`: Run benchmarks (see `just bench list` for options)
  - `just bench rust`: Rust core benchmarks for performance validation
  - `just bench collection`: End-to-end processing benchmarks (accepts scale parameter)
    - `just bench collection`: Default N=1 (1M entities)
    - `just bench collection 2`: N=2 (2M entities)
    - `just bench collection 0.5`: N=0.5 (500k entities)
    - N parameter scales all benchmark sizes proportionally

## Testing strategy

The project uses a **two-layer testing approach** that achieves comprehensive coverage without PyO3 linking complications:

**Layer 1: Pure Rust core tests** (33 tests)
- All business logic tested in `starlings-core` crate
- Zero PyO3 dependencies, no linking issues
- Full coverage of data structures, algorithms, and edge cases
- Run via: `cargo test -p starlings-core`

**Layer 2: Python integration tests** (17 tests)
- End-to-end testing of Python → Rust → Python data flow
- Tests PyO3 wrapper functionality, type conversions, error handling
- Validates real-world usage patterns and API contracts
- Run via: `uv run pytest`

This **"Rust Core with Python Bindings"** pattern ensures complete test coverage whilst avoiding complex PyO3 test configuration. The Rust core handles all business logic testing, whilst Python tests validate the integration boundary.

### Benchmarking and performance

Use `just bench` to run benchmarks for performance validation:
- `just bench rust`: Rust core benchmarks (hierarchy construction, 1k-10k edges) 
- `just bench collection`: End-to-end processing benchmarks (scaling tests, 1M+ edge datasets)
- Quantisation effect measurement
- Production-scale performance testing with PGO builds

**Performance achievements**:
- **5.4x improvement** over original implementation through key interning optimisations
- **Lock-free DataContext** using boxcar::Vec eliminates RwLock contention
- **Parallel processing** with rayon for multi-core scaling
- **PGO builds** achieve 645k edges/second throughput (1M edges in 1.6s)

There is a house style for parameterising Python unit tests:

```python
@pytest.mark.parametrize(
    ["foo", "bar"],
    [
        pytest.param(True, 12, id="test_thing"),
        pytest.param(False, 16, id="test_other_thing"),
    ],
)
def test_something(foo: bool, bar: int):
    """Tests that something does something."""
```

## Development workflow

1. Install dependencies: `just install`
2. Build the project: `just build` (or `just build pgo` for production performance)
3. Run tests: `just test`
4. Check formatting/linting: `just format`
5. Run benchmarks: `just bench collection` (for full performance testing)

### Performance builds

For production or performance testing, use Profile-Guided Optimisation:
- `just build pgo`: Builds with instrumentation, runs benchmarks, rebuilds optimised binary
- Achieves 25x performance improvement over original implementation
- Automatically cleans up profile data after build

## Project structure notes

- All source code is contained within `src/` subdirectories
- Root directory contains only configuration files and documentation
- **Two-crate architecture**: Separates pure Rust logic from Python bindings
- `starlings-core`: Business logic, algorithms, data structures (pure Rust)
- `starlings-py`: Minimal PyO3 wrapper over starlings-core (Python bindings)
- Follows the **Polars pattern** for professional Rust/Python hybrid projects
- Rust workspace configuration allows for clean dependency management
- Architecture enables both Python usage and pure Rust library consumption

## Design Documentation

Comprehensive design documents are available in `docs/design/`:

- **overview.md**: High-level system overview and capabilities
- **principles.md**: Mathematical foundations and theoretical guarantees
- **algorithms.md**: Core algorithms and data structures
- **engine.md**: Rust implementation details
- **interface.md**: Python API specification
- **roadmap.md**: Detailed implementation plan with specific tasks

These documents provide the complete technical specification for implementing Starlings from scratch.

## API Design

The current API follows the Polars wrapper pattern, where Rust classes (PyCollection, PyPartition) are wrapped in Python classes:

```python
import starlings as sl

# Create collection from edges
edges = [
    ("record_1", "record_2", 0.95),
    ("record_2", "record_3", 0.85),
    ("record_4", "record_5", 0.75),
]
collection = sl.Collection.from_edges(edges)

# Get partition at specific threshold
partition = collection.at(0.8)
print(f"Entities: {len(partition.entities)}")
```

**Implementation Pattern**:
- `starlings-py` exports `PyCollection` and `PyPartition` classes from Rust
- Python `__init__.py` imports these as private classes: `from .starlings import Collection as PyCollection`
- Public Python API wraps Rust classes: `Collection` wraps `PyCollection`, `Partition` wraps `PyPartition`
- This enables Python-friendly APIs whilst maintaining Rust performance

## Writing style guide

When writing documentation (README, comments, etc.):

- Use sentence case for headings: "How to use this" not "How To Use This"
- Add a line break after headings for readability
- Keep writing simple, clear and engaging
- Avoid overly technical jargon when possible
- Write for developers who want to get things done quickly
- Use British English spelling (organised, optimised, colour, etc.)