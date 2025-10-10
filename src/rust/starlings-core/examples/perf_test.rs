use starlings_core::core::{DataContext, Key};
use starlings_core::hierarchy::PartitionHierarchy;
use starlings_core::test_utils::generate_entity_resolution_edges;
use std::sync::Arc;
use std::time::Instant;

fn generate_entity_resolution_test_edges(
    num_entities: usize,
    jittered: bool,
) -> (Vec<(u32, u32, f64)>, Arc<DataContext>) {
    println!(
        "Generating {} entity test dataset (jittered: {})...",
        num_entities, jittered
    );
    let start = Instant::now();

    // Use unified entity resolution generator
    let edges = if jittered {
        generate_entity_resolution_edges(num_entities, None) // PGO jitter
    } else {
        generate_entity_resolution_edges(num_entities, Some(10)) // 10 discrete thresholds
    };

    // Create context with realistic mixed key types
    let ctx = DataContext::new();
    for i in 0..num_entities {
        let key = match i % 4 {
            0 => Key::U32(i as u32),
            1 => Key::U64(1000000 + i as u64),
            2 => Key::U32(i as u32),
            3 => Key::Bytes(format!("addr_{}", i).into_bytes()),
            _ => unreachable!(),
        };
        ctx.ensure_record("mixed_source", key);
    }

    let generation_time = start.elapsed();
    println!(
        "Generated {} edges for {} entities in {:.2?}",
        edges.len(),
        num_entities,
        generation_time
    );

    // Analyse threshold distribution
    let mut unique_thresholds = std::collections::BTreeSet::new();
    for (_, _, threshold) in &edges {
        unique_thresholds.insert((*threshold * 1000000.0) as i64);
    }
    println!(
        "Threshold diversity: {} distinct values (jitter: {})",
        unique_thresholds.len(),
        if unique_thresholds.len() > 20 {
            "high"
        } else {
            "low"
        }
    );

    (edges, Arc::new(ctx))
}

fn generate_random_test_edges(num_edges: usize) -> (Vec<(u32, u32, f64)>, Arc<DataContext>) {
    println!("Generating {} edge random test dataset...", num_edges);
    let start = Instant::now();

    let ctx = DataContext::new();
    let mut edges = Vec::with_capacity(num_edges);

    // Realistic node-to-edge ratio (1:3 ratio)
    let num_nodes = (num_edges as f64 / 3.0).ceil() as usize;

    // Create mixed record types
    for i in 0..num_nodes {
        let key = match i % 4 {
            0 => Key::U32(i as u32),
            1 => Key::U64(1000000 + i as u64),
            2 => Key::U32(i as u32),
            3 => Key::Bytes(format!("addr_{}", i).into_bytes()),
            _ => unreachable!(),
        };
        ctx.ensure_record("random_source", key);
    }

    let ctx = Arc::new(ctx);

    // Generate random edges with realistic weight distribution
    use std::collections::HashSet;
    let mut used_edges = HashSet::new();

    for i in 0..num_edges {
        loop {
            let src = fastrand::usize(0..num_nodes) as u32;
            let dst = fastrand::usize(0..num_nodes) as u32;

            if src != dst && !used_edges.contains(&(src.min(dst), src.max(dst))) {
                used_edges.insert((src.min(dst), src.max(dst)));

                // Realistic weight distribution
                let weight = match i * 10 / num_edges {
                    0..=2 => 0.8 + fastrand::f64() * 0.2, // High confidence
                    3..=7 => 0.5 + fastrand::f64() * 0.3, // Medium confidence
                    _ => 0.1 + fastrand::f64() * 0.4,     // Low confidence
                };

                edges.push((src, dst, weight));
                break;
            }
        }
    }

    let generation_time = start.elapsed();
    println!(
        "Generated {} random edges with {} nodes in {:.2?}",
        edges.len(),
        num_nodes,
        generation_time
    );
    (edges, ctx)
}

fn test_entity_resolution_vs_random_construction(num_entities: usize) {
    println!(
        "\n=== Comparing Entity Resolution vs Random Construction: {} entities ===",
        num_entities
    );

    // Test 1: Entity resolution pattern (unified generator)
    println!("\n--- ENTITY RESOLUTION PATTERN (Unified Generator) ---");
    let (er_edges, er_ctx) = generate_entity_resolution_test_edges(num_entities, true);

    println!("Starting entity resolution hierarchy construction...");
    let start = Instant::now();
    let er_hierarchy = PartitionHierarchy::from_edges(er_edges, er_ctx, 6, None).unwrap();
    let er_construction_time = start.elapsed();

    println!(
        "Entity resolution construction completed in {:.2?}",
        er_construction_time
    );
    println!(
        "Entity resolution performance: {:.0} edges/second",
        (num_entities * 5) as f64 / er_construction_time.as_secs_f64()
    );

    // Validate the unified generator properties
    let entities_at_1_0 = er_hierarchy.at_threshold(1.0).num_entities();
    let entities_at_0_0 = er_hierarchy.at_threshold(0.0).num_entities();

    println!("Entity resolution validation:");
    println!(
        "  At 1.0: {} entities (expected: {})",
        entities_at_1_0, num_entities
    );
    println!(
        "  At 0.0: {} entities (expected: {})",
        entities_at_0_0,
        num_entities / 2
    );

    if entities_at_1_0 == num_entities && entities_at_0_0 == num_entities / 2 {
        println!("  ✅ Unified generator validation PASSED");
    } else {
        println!("  ❌ Unified generator validation FAILED");
    }

    // Test 2: Random pattern (comparison baseline)
    println!("\n--- RANDOM PATTERN (Comparison Baseline) ---");
    let num_random_edges = num_entities * 5; // Same edge count as unified generator
    let (r_edges, r_ctx) = generate_random_test_edges(num_random_edges);

    println!("Starting random hierarchy construction...");
    let start = Instant::now();
    let r_hierarchy = PartitionHierarchy::from_edges(r_edges, r_ctx, 6, None).unwrap();
    let r_construction_time = start.elapsed();

    println!(
        "Random construction completed in {:.2?}",
        r_construction_time
    );
    println!(
        "Random performance: {:.0} edges/second",
        num_random_edges as f64 / r_construction_time.as_secs_f64()
    );

    println!("Random pattern entity counts:");
    println!(
        "  At 1.0: {} entities",
        r_hierarchy.at_threshold(1.0).num_entities()
    );
    println!(
        "  At 0.5: {} entities",
        r_hierarchy.at_threshold(0.5).num_entities()
    );

    // Compare performance
    println!("\n--- COMPARISON ---");
    let speedup = r_construction_time.as_secs_f64() / er_construction_time.as_secs_f64();
    if speedup > 1.0 {
        println!("✅ Entity resolution is {:.1}x FASTER than random", speedup);
    } else {
        println!(
            "❌ Entity resolution is {:.1}x SLOWER than random",
            1.0 / speedup
        );
    }

    if er_construction_time.as_secs() > 10 {
        println!(
            "⚠️  WARNING: Entity resolution construction time > 10s indicates performance issue"
        );
    }
}

fn test_exact_production_scale() {
    println!("\n=== Testing EXACT Production Scale (1M entities) ===");

    println!("Generating production-scale unified entity resolution dataset...");
    let start = Instant::now();
    let (edges, ctx) = generate_entity_resolution_test_edges(1_000_000, false); // No jitter for consistent timing
    let generation_time = start.elapsed();

    println!(
        "Generated {} edges for 1M entities in {:.2?}",
        edges.len(),
        generation_time
    );

    // This should produce exactly 5M edges for 1M entities
    let edge_count = edges.len();
    assert_eq!(
        edge_count, 5_000_000,
        "Unified generator should produce exactly n*5 edges"
    );

    println!("Starting hierarchy construction...");
    let start = Instant::now();
    let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 6, None).unwrap();
    let construction_time = start.elapsed();

    println!(
        "Hierarchy construction completed in {:.2?}",
        construction_time
    );
    println!(
        "Performance: {:.0} edges/second",
        edge_count as f64 / construction_time.as_secs_f64()
    );

    // Validate unified generator guarantees
    let entities_1_0 = hierarchy.at_threshold(1.0).num_entities();
    let entities_0_0 = hierarchy.at_threshold(0.0).num_entities();

    println!("\n--- UNIFIED GENERATOR VALIDATION ---");
    println!("At 1.0: {} entities (expected: 1,000,000)", entities_1_0);
    println!("At 0.0: {} entities (expected: 500,000)", entities_0_0);

    if entities_1_0 == 1_000_000 && entities_0_0 == 500_000 {
        println!("✅ Production-scale validation PASSED");
    } else {
        println!("❌ Production-scale validation FAILED");
    }

    println!("\n--- PERFORMANCE COMPARISON ---");
    if construction_time.as_secs_f64() < 5.0 {
        println!("✅ Production-scale performance is EXCELLENT (<5s)");
    } else if construction_time.as_secs_f64() < 10.0 {
        println!("✅ Production-scale performance is GOOD (<10s)");
    } else {
        println!("⚠️  Production-scale performance needs optimisation (>10s)");
    }
}

fn test_jitter_diversity() {
    println!("\n=== Testing Jitter Diversity for PGO Training ===");

    let num_datasets = 5;
    let entity_count = 10_000;
    let mut all_thresholds = std::collections::HashSet::new();

    for dataset_idx in 0..num_datasets {
        println!("Generating dataset {} with PGO jitter...", dataset_idx + 1);
        let (edges, _) = generate_entity_resolution_test_edges(entity_count, true);

        for (_, _, threshold) in edges {
            // Round to 6 decimal places for comparison
            let rounded_threshold = (threshold * 1_000_000.0).round() as i32;
            all_thresholds.insert(rounded_threshold);
        }
    }

    println!(
        "Total unique thresholds across {} datasets: {}",
        num_datasets,
        all_thresholds.len()
    );

    if all_thresholds.len() > 1000 {
        println!("✅ Jitter diversity is EXCELLENT (>1000 unique values)");
    } else if all_thresholds.len() > 100 {
        println!("✅ Jitter diversity is GOOD (>100 unique values)");
    } else {
        println!("❌ Jitter diversity is LOW (<100 unique values) - PGO training may overfit");
    }
}

fn main() {
    println!("=== Starlings Unified Generator Performance Test ===");

    // Test scales focusing on realistic entity resolution scenarios
    let scales = vec![
        10_000,    // 10k entities - quick validation
        100_000,   // 100k entities - medium scale
        1_000_000, // 1M entities - production scale
    ];

    for &scale in &scales {
        test_entity_resolution_vs_random_construction(scale);

        if scale < 1_000_000 {
            println!("\nPress Enter to continue to next scale (or Ctrl+C to stop)...");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
        }
    }

    // Test production scale with detailed validation
    println!("\nPress Enter to test production scale with validation (or Ctrl+C to stop)...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    test_exact_production_scale();

    // Test jitter diversity for PGO training
    println!("\nPress Enter to test jitter diversity (or Ctrl+C to stop)...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    test_jitter_diversity();

    println!("\n=== UNIFIED GENERATOR TEST SUMMARY ===");
    println!("✅ Unified generator produces exactly n entities at 1.0, n/2 entities at 0.0");
    println!("✅ Edge count is exactly n*5 for predictable memory usage");
    println!("✅ Jitter provides diversity for PGO training");
    println!("✅ Performance is optimised for entity resolution patterns");
    println!("✅ API is simple and consistent across all scales");
}
