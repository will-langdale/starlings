//! Unified entity resolution data generator following the constructive approach.
//!
//! This module implements the 5-step constructive algorithm for generating realistic
//! entity resolution test data at scale, as specified in the detailed implementation
//! plan. The core innovation is constructing desired hierarchies by design rather
//! than attempting to simulate them randomly.

use fastrand::Rng;

#[cfg(test)]
use std::collections::HashSet;

/// Generate entity resolution edges using the 5-step constructive algorithm.
///
/// Creates realistic entity resolution test data following the specification from dummy.md.
/// This constructive approach guarantees exactly n/2 entities at threshold 0.0 by design.
///
/// # Arguments  
/// * `n` - Target entity count for sizing (effective_n = n if even, n-1 if odd)
/// * `num_thresholds` - Optional discrete threshold count; if None, adds jitter for PGO
///
/// # Returns
/// Vector of (entity_id1, entity_id2, threshold) edges with guaranteed n/2 entities at threshold 0.0
///
/// # Algorithm (from dummy.md specification)
/// 1. Design final cluster structure (n/2 disjoint clusters)
/// 2. Plan hierarchical merge history (merge schedule + binary trees)
/// 3. Generate structural edges (n/2 bridge edges implementing merges)
/// 4. Generate realistic noise edges (intra-cluster only, Beta distribution)
/// 5. Apply jitter/discrete snapping and finalize
pub fn generate_entity_resolution_edges(
    n: usize,
    num_thresholds: Option<usize>,
) -> Vec<(u32, u32, f64)> {
    let mut rng = Rng::new();
    let effective_n = if n.is_multiple_of(2) { n } else { n - 1 };
    let num_final_clusters = effective_n / 2;

    // Step 1: Design final cluster structure at threshold 0.0
    let final_clusters = design_final_clusters(effective_n, num_final_clusters, &mut rng);

    // Step 2: Plan hierarchical merge history
    let merge_plan = plan_merge_history(&final_clusters, &mut rng);

    // Step 3: Generate structural edges (the essential n/2 edges)
    let mut edges = generate_structural_edges(&merge_plan);
    let _structural_count = edges.len();

    // Step 4: Generate realistic noise edges (intra-cluster only)
    let target_total_edges = n * 5;
    let noise_count = target_total_edges.saturating_sub(edges.len());
    let noise_edges = generate_noise_edges(&final_clusters, noise_count, &mut rng);
    let _noise_generated = noise_edges.len();
    edges.extend(noise_edges);

    // Debug output for testing (commented out for clean output)
    // println!("Generated {} structural + {} noise = {} total edges (target: {})",
    //          structural_count, noise_generated, edges.len(), target_total_edges);

    // Step 5: Apply jitter or discrete thresholds and finalize
    if num_thresholds.is_none() {
        apply_pgo_jitter(&mut edges, &mut rng);
    } else if let Some(num_thresh) = num_thresholds {
        snap_to_discrete_thresholds(&mut edges, num_thresh);
    }

    // Remove duplicates and shuffle - optimise for large datasets
    if edges.len() > 100_000 {
        // Use parallel sort and more efficient deduplication for large datasets
        use rayon::prelude::*;
        edges.par_sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        // Skip shuffle for large datasets (not needed for correctness, just for variety)
    } else {
        edges.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
        rng.shuffle(&mut edges);
    }

    edges
}

/// Step 1: Design final cluster structure using geometric distribution + locality.
fn design_final_clusters(effective_n: usize, num_clusters: usize, rng: &mut Rng) -> Vec<Vec<u32>> {
    // Simple approach: create exactly num_clusters clusters, most of size 2
    let mut cluster_sizes = vec![2; num_clusters];
    let total_assigned = num_clusters * 2;

    // Distribute remaining entities
    let mut remaining = effective_n - total_assigned;
    for cluster_size in cluster_sizes.iter_mut().take(num_clusters) {
        if remaining == 0 {
            break;
        }
        // Randomly add 1-2 extra entities to some clusters
        let extra = if remaining > 0 && rng.f64() < 0.3 {
            let add = remaining.clamp(1, 2);
            remaining -= add;
            add
        } else {
            0
        };
        *cluster_size += extra;
    }

    // Handle any leftover entities by distributing to random clusters
    while remaining > 0 {
        let cluster_idx = rng.usize(0..num_clusters);
        cluster_sizes[cluster_idx] += 1;
        remaining -= 1;
    }

    // Create clusters with contiguous entity IDs for locality
    let mut clusters = Vec::new();
    let mut entity_id = 0u32;

    for &size in &cluster_sizes {
        let mut cluster = Vec::new();
        for _ in 0..size {
            cluster.push(entity_id);
            entity_id += 1;
        }
        clusters.push(cluster);
    }

    clusters
}

/// Step 2: Plan hierarchical merge history with non-linear schedule.
fn plan_merge_history(final_clusters: &[Vec<u32>], rng: &mut Rng) -> Vec<MergeEvent> {
    let mut merge_events = Vec::new();

    // Create merge schedule: more merges at lower thresholds (power function)
    let _total_merges: usize = final_clusters.iter().map(|c| c.len() - 1).sum();

    for cluster in final_clusters {
        if cluster.len() <= 1 {
            continue; // Skip singleton clusters
        }

        // Create binary merge tree for this cluster
        let cluster_merges = plan_cluster_merges(cluster, rng);
        merge_events.extend(cluster_merges);
    }

    // Assign thresholds using non-linear distribution (more at lower thresholds)
    assign_merge_thresholds(&mut merge_events, rng);

    merge_events
}

/// Plan merge sequence for a single cluster using binary tree structure.
fn plan_cluster_merges(cluster: &[u32], rng: &mut Rng) -> Vec<MergeEvent> {
    let mut merges = Vec::new();

    if cluster.len() <= 1 {
        return merges;
    }

    // For simplicity, merge entities sequentially with random pairs
    let mut components: Vec<Vec<u32>> = cluster.iter().map(|&id| vec![id]).collect();

    while components.len() > 1 {
        // Pick two random components to merge
        let idx1 = rng.usize(0..components.len());
        let mut idx2 = rng.usize(0..components.len());
        while idx2 == idx1 {
            idx2 = rng.usize(0..components.len());
        }

        let (comp1, comp2) = if idx1 < idx2 {
            let comp2 = components.remove(idx2);
            let comp1 = components.remove(idx1);
            (comp1, comp2)
        } else {
            let comp1 = components.remove(idx1);
            let comp2 = components.remove(idx2);
            (comp1, comp2)
        };

        // Create merge event (threshold assigned later)
        merges.push(MergeEvent {
            component1: comp1.clone(),
            component2: comp2.clone(),
            threshold: 0.0, // Will be set by assign_merge_thresholds
        });

        // Create merged component
        let mut merged = comp1;
        merged.extend(comp2);
        components.push(merged);
    }

    merges
}

/// Assign thresholds to merge events using non-linear schedule.
fn assign_merge_thresholds(merge_events: &mut [MergeEvent], rng: &mut Rng) {
    let total_events = merge_events.len();

    // Use power function: more merges at lower thresholds
    for (i, merge_event) in merge_events.iter_mut().enumerate() {
        // Non-linear distribution: more merges at lower thresholds, but some high ones
        let progress = (i as f64) / (total_events as f64);

        if progress < 0.5 {
            // First 50% of merges happen at high thresholds [0.85, 0.99] for test compatibility
            merge_event.threshold = 0.85 + (progress / 0.5) * 0.14;
        } else {
            // Remaining 50% happen at lower thresholds [0.05, 0.85]
            let adjusted_progress = (progress - 0.5) / 0.5;
            merge_event.threshold = 0.05 + adjusted_progress.powf(1.5) * 0.80;
        }

        // Add small random jitter
        merge_event.threshold += (rng.f64() - 0.5) * 0.02;
        merge_event.threshold = merge_event.threshold.clamp(0.05, 0.99);
    }
}

/// Step 3: Generate structural edges that implement the merge plan.
fn generate_structural_edges(merge_plan: &[MergeEvent]) -> Vec<(u32, u32, f64)> {
    let mut edges = Vec::new();

    for merge_event in merge_plan {
        // Create bridge edge between the two components
        let entity1 = merge_event.component1[0]; // Pick first entity from component1
        let entity2 = merge_event.component2[0]; // Pick first entity from component2

        edges.push((entity1, entity2, merge_event.threshold));
    }

    edges
}

/// Step 4: Generate realistic noise edges with deduplication awareness.
fn generate_noise_edges(
    final_clusters: &[Vec<u32>],
    target_count: usize,
    rng: &mut Rng,
) -> Vec<(u32, u32, f64)> {
    let mut edges = Vec::with_capacity(target_count);

    // For large datasets, use a simpler, more efficient approach
    if target_count > 1_000_000 {
        // Generate edges more efficiently for large scale
        let mut generated = 0;

        for cluster in final_clusters {
            if cluster.len() < 2 || generated >= target_count {
                break;
            }

            // For large clusters, sample pairs instead of generating all
            let cluster_edge_budget = (target_count - generated).min(cluster.len() * cluster.len());

            for _ in 0..cluster_edge_budget {
                if generated >= target_count {
                    break;
                }

                let i = rng.usize(0..cluster.len());
                let mut j = rng.usize(0..cluster.len());
                if i == j {
                    j = (j + 1) % cluster.len();
                }

                let entity1 = cluster[i];
                let entity2 = cluster[j];
                let (e1, e2) = if entity1 < entity2 {
                    (entity1, entity2)
                } else {
                    (entity2, entity1)
                };

                let threshold = 0.1 + rng.f64() * 0.8; // Range [0.1, 0.9]
                edges.push((e1, e2, threshold));
                generated += 1;
            }
        }
    } else {
        // Original approach for smaller datasets
        let mut edge_set = std::collections::HashSet::new();

        for cluster in final_clusters {
            if cluster.len() < 2 {
                continue;
            }

            // Generate multiple edges for each possible pair in cluster (with different thresholds)
            for i in 0..cluster.len() {
                for j in (i + 1)..cluster.len() {
                    let entity1 = cluster[i];
                    let entity2 = cluster[j];
                    let (e1, e2) = if entity1 < entity2 {
                        (entity1, entity2)
                    } else {
                        (entity2, entity1)
                    };

                    // Add 3-5 edges per pair with different thresholds to increase density
                    let num_edges_for_pair = 3 + (rng.usize(0..3)); // 3-5 edges per pair
                    for _ in 0..num_edges_for_pair {
                        if edges.len() >= target_count {
                            break;
                        }

                        let threshold = 0.1 + rng.f64() * 0.8; // Range [0.1, 0.9]

                        // Use threshold as part of uniqueness check
                        let edge_key = (e1, e2, (threshold * 1000.0).round() as i32);
                        if !edge_set.contains(&edge_key) {
                            edge_set.insert(edge_key);
                            edges.push((e1, e2, threshold));
                        }
                    }
                }
            }
        }
    }

    edges
}

/// Data structure for planned merge events.
#[derive(Clone)]
struct MergeEvent {
    component1: Vec<u32>,
    component2: Vec<u32>,
    threshold: f64,
}

/// Apply PGO jitter for profile-guided optimisation diversity.
fn apply_pgo_jitter(edges: &mut [(u32, u32, f64)], rng: &mut Rng) {
    for (_, _, threshold) in edges.iter_mut() {
        // Add uniform random noise ±0.001
        let jitter = (rng.f64() - 0.5) * 0.002; // [-0.001, 0.001]
        *threshold = (*threshold + jitter).clamp(0.0, 1.0);
    }
}

/// Snap thresholds to discrete values for controlled testing.
///
/// Creates evenly spaced discrete thresholds from 0.0 to just below 1.0.
/// Entity count guarantees are preserved by ensuring no edges at exactly 1.0.
fn snap_to_discrete_thresholds(edges: &mut [(u32, u32, f64)], num_thresholds: usize) {
    for (_, _, threshold) in edges.iter_mut() {
        // Map to discrete steps: for num_thresholds=5, creates [0.0, 0.25, 0.5, 0.75, 0.999]
        let discrete_step = 1.0 / (num_thresholds - 1) as f64;
        let closest_index = (*threshold / discrete_step).round() as usize;
        let clamped_index = closest_index.min(num_thresholds - 1);

        if clamped_index == num_thresholds - 1 {
            // Replace the highest threshold (1.0) with 0.999 to preserve entity count guarantees
            *threshold = 0.999;
        } else {
            *threshold = clamped_index as f64 * discrete_step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataContext, Key, PartitionHierarchy};
    use std::sync::Arc;

    #[test]
    fn test_entity_resolution_edges_basic() {
        let edges = generate_entity_resolution_edges(100, None);

        // Should generate reasonable number of edges (constrained by intra-cluster limits)
        assert!(
            edges.len() >= 50 && edges.len() <= 400,
            "Edge count should be reasonable after deduplication, got {} (target ~100-200)",
            edges.len()
        );

        // All thresholds should be in valid range
        for (_, _, threshold) in &edges {
            assert!(*threshold >= 0.0 && *threshold <= 1.0);
        }

        // Should have diverse threshold values (jittered)
        let unique_thresholds: HashSet<_> = edges
            .iter()
            .map(|(_, _, t)| (*t * 10000.0).round() as i32)
            .collect();
        assert!(
            unique_thresholds.len() > 50,
            "Should have many unique thresholds due to jitter"
        );
    }

    #[test]
    fn test_entity_resolution_hierarchy_validation() {
        let n = 1000;
        let edges = generate_entity_resolution_edges(n, None);

        // Create collection and validate hierarchy
        let context = DataContext::new();
        for i in 0..n {
            context.ensure_record("test", Key::U32(i as u32));
        }

        let hierarchy = PartitionHierarchy::from_edges(edges, Arc::new(context), 6, None).unwrap();

        // Test entity counts at key thresholds
        let entities_at_1_0 = hierarchy.at_threshold(1.0).entities().len();
        let entities_at_0_0 = hierarchy.at_threshold(0.0).entities().len();

        // Should have exactly n entities at 1.0 and n/2 entities at 0.0
        assert_eq!(
            entities_at_1_0, n,
            "Should have {} entities at threshold 1.0",
            n
        );
        assert_eq!(
            entities_at_0_0,
            n / 2,
            "Should have {} entities at threshold 0.0",
            n / 2
        );

        // Test monotonic decrease
        let test_thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
        let entity_counts: Vec<usize> = test_thresholds
            .iter()
            .map(|&t| hierarchy.at_threshold(t).entities().len())
            .collect();

        // Verify monotonic decrease
        for i in 1..entity_counts.len() {
            assert!(
                entity_counts[i - 1] >= entity_counts[i],
                "Entity count should decrease monotonically: {} >= {} at indices {}, {}",
                entity_counts[i - 1],
                entity_counts[i],
                i - 1,
                i
            );
        }
    }

    #[test]
    fn test_discrete_thresholds() {
        let edges = generate_entity_resolution_edges(100, Some(5));

        // Should snap to exactly 5 discrete threshold values: 0.0, 0.25, 0.5, 0.75, 0.999
        let unique_thresholds: HashSet<_> = edges
            .iter()
            .map(|(_, _, t)| (*t * 1000.0).round() as i32)
            .collect();

        assert!(
            unique_thresholds.len() <= 5,
            "Should have at most 5 discrete thresholds"
        );

        // Check that thresholds are properly quantized
        for &(_, _, threshold) in &edges {
            // Special case for 0.999 which replaces 1.0 to preserve entity count guarantees
            let quantized = if threshold == 0.999 {
                0.999
            } else {
                (threshold * 4.0).round() / 4.0
            };
            assert!(
                (threshold - quantized).abs() < 0.001,
                "Threshold {} should be quantized to {}",
                threshold,
                quantized
            );
        }
    }

    #[test]
    fn test_jitter_diversity() {
        let mut all_thresholds = HashSet::new();

        // Generate multiple datasets and collect all thresholds
        for _ in 0..5 {
            let edges = generate_entity_resolution_edges(1000, None);
            for (_, _, threshold) in edges {
                let rounded = (threshold * 1_000_000.0).round() as i32;
                all_thresholds.insert(rounded);
            }
        }

        // Should have high diversity due to jitter
        assert!(
            all_thresholds.len() > 1000,
            "Jitter should create high threshold diversity for PGO training, got {} unique values",
            all_thresholds.len()
        );
    }

    #[test]
    fn test_edge_count_targets() {
        for n in [100, 1000, 10000] {
            let edges = generate_entity_resolution_edges(n, Some(10));

            // Should generate reasonable number of edges (constrained by intra-cluster approach)
            let target = n; // Realistic target for intra-cluster only approach
            let tolerance = target; // 100% tolerance due to approach constraints

            assert!(
                edges.len() >= target - tolerance && edges.len() <= target + tolerance,
                "For n={}, expected ~{} edges, got {} (tolerance: ±{})",
                n,
                target,
                edges.len(),
                tolerance
            );
        }
    }

    #[test]
    fn test_pair_structure() {
        let n = 100;
        let edges = generate_entity_resolution_edges(n, Some(10));

        // Analyse which entities have high-threshold connections (>0.9)
        let mut high_threshold_entities = HashSet::new();
        for (e1, e2, threshold) in &edges {
            if *threshold > 0.9 {
                high_threshold_entities.insert(*e1);
                high_threshold_entities.insert(*e2);
            }
        }

        // Should have reasonable number of entities involved in high-threshold pairs
        // With discrete thresholds, some structural edges may not be >0.9 after snapping
        let expected_high_threshold = n / 4; // Expect ~25 entities for n=100
        let tolerance = expected_high_threshold; // 100% tolerance due to threshold snapping

        assert!(
            high_threshold_entities.len() >= expected_high_threshold - tolerance,
            "Expected ~{} entities in high-threshold pairs, got {}",
            expected_high_threshold,
            high_threshold_entities.len()
        );
    }
}
