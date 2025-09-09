# Unit testing
mod test 'src/tests/justfile'
# Benchmarking  
mod bench 'src/tests/benchmarks/justfile'
# Building
mod build 'src/justfile'

# Environment variables
set export
STARLINGS_DEBUG := "1"

# List all commands
default:
    just --list
    @echo "just test"
    just test list
    @echo "just bench"
    just bench list
    @echo "just build"
    just build list

# Install development dependencies
install:
    uv sync

# Format and lint all code (Python + Rust)
format:
    uvx ruff check src/ --fix
    uvx ruff format src/
    uvx --with pip mypy src/python/ --install-types --non-interactive
    cargo fmt
    cargo clippy --all-targets --all-features

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/
    find src/ -name "*.pyc" -delete
    find src/ -name "__pycache__" -delete

# Run a local documentation development server
docs:
    uv run mkdocs serve
