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

The project uses a **three-layer testing approach** that achieves comprehensive coverage whilst maintaining system safety:

**Layer 1: Pure Rust core tests** (69 tests)
- All business logic tested in `starlings-core` crate
- Zero PyO3 dependencies, no linking issues
- Full coverage of data structures, algorithms, and edge cases
- Run via: `cargo test -p starlings-core`

**Layer 2: Python integration tests** (19 tests)
- Unit tests with small datasets (< 1000 entities) for basic functionality
- E2E tests with production-scale datasets (100k-1M entities) for safety validation
- Tests PyO3 wrapper functionality, type conversions, error handling
- **SAFE**: All tests run with `STARLINGS_MEMORY_LIMIT=50%`
- Run via: `just test` (safe by default with production validation)
- **IMPORTANT**: When running Python tests directly (not via `just`), always use `uv run pytest` to ensure correct virtual environment

**Layer 3: Stress testing** (12+ tests)  
- Memory pressure simulation, resource exhaustion scenarios
- Circuit breaker validation under artificial system stress
- **DANGEROUS**: Artificially consumes system resources to test limits
- Run via: `just test dangerous` (explicit opt-in required)

This **"Rust Core with Python Bindings"** pattern ensures complete test coverage whilst avoiding complex PyO3 test configuration. The Rust core handles all business logic testing, whilst Python tests validate the integration boundary.

**Safety by Default**: The default `just test` command runs safe tests including production-scale validation that demonstrate the safety system working correctly. The safety system automatically prevents system crashes during large-scale operations. Only stress tests that artificially exhaust system resources require explicit opt-in via `just test dangerous`.

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

**Code Quality Standards**:
- All code follows British English spelling and conventions (comments, docs, variable names)
- Note: External crate methods like `.serialized_size()` use American spelling - this is expected
- Comprehensive type annotations and docstrings following Google style
- DRY principles with extracted helper methods for common patterns
- Context managers for resource management (environment variables, etc.)
- Consistent error handling and validation patterns
- Use `debug_println!` macro for all debug output, never direct `eprintln!` for user-facing messages

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

## Environment Variables

Starlings supports several environment variables for configuration:

- **`STARLINGS_MEMORY_LIMIT`**: Memory limit for all operations (default: 80% of RAM)
  - Controls the maximum memory Starlings will use
  - Can be specified as:
    - Percentage: `50%`, `80%` (percentage of total system RAM)
    - Absolute size: `10GB`, `4096MB`, `4096` (MB assumed if no unit)
  - The partition cache has its own memory bounds (25% of total limit) with LRU eviction
  - Cache memory is managed independently from operation memory limits
  - This follows the DuckDB model where cache and query memory are separate
  - Example: `STARLINGS_MEMORY_LIMIT=10GB` limits total usage to 10GB

- **`STARLINGS_DEBUG`**: Enable debug output (0 or 1, default: 0)
  - Set to 1 to enable detailed debug information during processing

## Memory Spilling Architecture

Starlings uses a simplified two-tier storage strategy with optional spilling for large operations. This enables safe processing of large datasets whilst maintaining excellent performance for typical workloads.

### Storage Backends

**Hierarchy Storage** (2 options):
- **InMemoryStorage**: Fast, for datasets <50% of memory limit
- **DiskStorage**: Disk-backed with LRU cache, for larger datasets

Selection is automatic based on estimated size:
```rust
if estimated_mb < memory_limit_mb / 2 {
    InMemoryStorage  // Fast path
} else {
    DiskStorage      // Large datasets (10M+ edges)
}
```

### Spillable Operations

Large operations can implement the `Spillable` trait for automatic memory management:

```rust
use starlings_core::core::spilling::{Spillable, SpillHandle, SpillError};

impl Spillable for MyLargeOperation {
    fn estimated_memory_bytes(&self) -> u64 { /* */ }
    fn spill_to_disk(&mut self) -> Result<SpillHandle, SpillError> { /* */ }
    fn restore_from_disk(&mut self, handle: SpillHandle) -> Result<(), SpillError> { /* */ }
    fn is_spilled(&self) -> bool { /* */ }
}
```

**When to spill**:
- <100MB: Keep in memory (fast, simple)
- 100-500MB: Check if fits in limit, spill if needed
- >500MB: Always spill or stream

**Current spillable operations**:
- Delta algorithm state (auto-spills at 200MB)
- Hierarchy storage (via DiskStorage backend)
- Partition cache (self-manages with LRU eviction)

### Adding Spillable Operations

When adding memory-intensive features:

1. **Estimate memory** before allocation:
   ```rust
   let estimated_mb = calculate_memory_needed();
   ensure_memory_safety(estimated_mb)?;
   ```

2. **Implement Spillable** if operation >100MB possible:
   ```rust
   impl Spillable for MyOperation {
       fn estimated_memory_bytes(&self) -> u64 {
           // Calculate actual usage
       }

       fn spill_to_disk(&mut self) -> Result<SpillHandle, SpillError> {
           // Serialize to temp file via create_spill_file()
           // Clear in-memory data
           // Return handle
       }

       fn restore_from_disk(&mut self, handle: SpillHandle) -> Result<(), SpillError> {
           // Deserialize from handle.path
           // Restore in-memory state
       }
   }
   ```

3. **Document behavior** in operation's docstring:
   ```rust
   /// Large operation that auto-spills at 200MB.
   ///
   /// Spilling is transparent - operation continues normally
   /// but may be slower when accessing spilled data.
   ```

See `src/rust/starlings-core/src/core/spilling.rs` for complete trait definition and utilities.

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