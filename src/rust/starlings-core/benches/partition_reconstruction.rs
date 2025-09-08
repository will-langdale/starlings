use criterion::{black_box, criterion_group, criterion_main, Criterion};
use starlings_core::core::{DataContext, Key};
use starlings_core::hierarchy::PartitionHierarchy;
use starlings_core::test_utils::generate_entity_resolution_edges;
use std::sync::Arc;

fn generate_test_hierarchy(entity_count: usize) -> PartitionHierarchy {
    let ctx = DataContext::new();

    // Create mixed record types for realistic benchmarking
    for i in 0..entity_count {
        match i % 4 {
            0 => ctx.ensure_record("customers", Key::String(format!("cust_{}", i))),
            1 => ctx.ensure_record("transactions", Key::U64(1000000 + i as u64)),
            2 => ctx.ensure_record("products", Key::U32(i as u32)),
            3 => ctx.ensure_record("addresses", Key::Bytes(format!("addr_{}", i).into_bytes())),
            _ => unreachable!(),
        };
    }

    let ctx = Arc::new(ctx);

    // Use unified entity resolution generator for realistic patterns
    let edges = generate_entity_resolution_edges(entity_count, Some(10)); // Discrete thresholds for consistent timing

    println!(
        "Built hierarchy with {} edges and {} records using unified generator",
        edges.len(),
        ctx.len()
    );
    PartitionHierarchy::from_edges(edges, ctx, 6, None).unwrap()
}

fn bench_partition_reconstruction_1m(c: &mut Criterion) {
    // Use 200k entities which produces ~1M edges for realistic benchmarking
    let mut hierarchy = generate_test_hierarchy(200_000);

    let mut group = c.benchmark_group("partition_reconstruction_production");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("200k_entities_threshold_access", |b| {
        b.iter(|| {
            // Test uncached access to different thresholds for realistic entity resolution patterns
            black_box(hierarchy.at_threshold(0.95));
            black_box(hierarchy.at_threshold(0.85));
            black_box(hierarchy.at_threshold(0.75));
        })
    });

    group.finish();
}

criterion_group!(benches, bench_partition_reconstruction_1m);
criterion_main!(benches);
