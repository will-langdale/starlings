//! Spillable trait and infrastructure for memory-intensive operations
//!
//! This module provides a standard pattern for operations that can spill
//! to disk when memory pressure is high. It includes the core trait,
//! error types, and utilities for managing spilled data.

use std::io;
use std::path::PathBuf;

/// Trait for data structures that can spill to disk when memory constrained
pub trait Spillable: Send + Sync {
    /// Estimate current memory usage in bytes
    fn estimated_memory_bytes(&self) -> u64;

    /// Spill contents to disk, returning a handle for restoration
    ///
    /// After spilling, the in-memory representation should be cleared
    /// to free memory. The returned handle allows restoration later.
    fn spill_to_disk(&mut self) -> Result<SpillHandle, SpillError>;

    /// Restore contents from disk using a handle
    ///
    /// Recreates the in-memory representation from spilled data.
    fn restore_from_disk(&mut self, handle: SpillHandle) -> Result<(), SpillError>;

    /// Check if currently spilled to disk
    fn is_spilled(&self) -> bool;
}

/// Handle to spilled data on disk
///
/// Manages the lifecycle of temporary spill files. Files are
/// automatically cleaned up when the handle is dropped.
#[derive(Debug, Clone)]
pub struct SpillHandle {
    /// Path to spilled data file
    pub path: PathBuf,
    /// Size of spilled data in bytes
    pub size_bytes: u64,
}

impl Drop for SpillHandle {
    fn drop(&mut self) {
        // Best effort cleanup - ignore errors
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Errors that can occur during spilling operations
#[derive(Debug, thiserror::Error)]
pub enum SpillError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Already spilled to disk")]
    AlreadySpilled,

    #[error("Not spilled - cannot restore")]
    NotSpilled,
}

/// Decision: should this operation spill to disk?
///
/// Based on estimated memory usage and available memory limit.
pub fn should_spill(estimated_mb: u64, memory_limit_mb: u64) -> bool {
    // Spill if estimated usage exceeds 50% of limit
    estimated_mb > memory_limit_mb / 2
}

/// Create a temporary file for spilling data
///
/// Returns a NamedTempFile that will be cleaned up on drop.
pub fn create_spill_file(prefix: &str) -> Result<tempfile::NamedTempFile, SpillError> {
    tempfile::Builder::new()
        .prefix(prefix)
        .suffix(".spill")
        .tempfile()
        .map_err(SpillError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_spill_logic() {
        // Small operation: don't spill
        assert!(!should_spill(100, 1000)); // 10% of limit

        // Medium operation: don't spill if under 50%
        assert!(!should_spill(400, 1000)); // 40% of limit

        // Large operation: spill if over 50%
        assert!(should_spill(600, 1000)); // 60% of limit
        assert!(should_spill(800, 1000)); // 80% of limit
    }

    #[test]
    fn test_spill_handle_cleanup() -> Result<(), SpillError> {
        let temp_file = create_spill_file("test")?;
        let path = temp_file.path().to_path_buf();

        // Create handle (takes ownership of temp file path)
        let handle = SpillHandle {
            path: path.clone(),
            size_bytes: 100,
        };

        // File should exist while handle exists
        drop(temp_file);

        // Drop handle - file should be cleaned up
        drop(handle);

        Ok(())
    }
}
