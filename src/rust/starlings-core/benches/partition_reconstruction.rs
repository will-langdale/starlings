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
            0 => ctx.ensure_record("customers", Key::U32(i as u32)),
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
    let hierarchy = generate_test_hierarchy(200_000);

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

fn bench_sweep_reconstruction(c: &mut Criterion) {
    // Test incremental reconstruction for sweep operations

    let mut group = c.benchmark_group("sweep_reconstruction");
    group.sample_size(10);

    // Benchmark naive approach (individual at_threshold calls) - with fresh hierarchy each time
    group.bench_function("naive_11_thresholds_uncached", |b| {
        b.iter(|| {
            // Create fresh hierarchy to avoid cache hits
            let hierarchy = generate_test_hierarchy(50_000);
            let thresholds: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
            let partitions: Vec<_> = thresholds
                .iter()
                .map(|&t| hierarchy.at_threshold(t))
                .collect();
            black_box(partitions)
        })
    });

    // Benchmark incremental approach - also with fresh hierarchy to be fair
    group.bench_function("incremental_11_thresholds_uncached", |b| {
        b.iter(|| {
            let hierarchy = generate_test_hierarchy(50_000);
            let thresholds: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
            let partitions = hierarchy.build_partitions_incrementally(&thresholds);
            black_box(partitions)
        })
    });

    // Benchmark with larger sweep (21 thresholds) - incremental only
    group.bench_function("incremental_21_thresholds_uncached", |b| {
        b.iter(|| {
            let hierarchy = generate_test_hierarchy(50_000);
            let thresholds: Vec<f64> = (0..=20).map(|i| i as f64 / 20.0).collect();
            let partitions = hierarchy.build_partitions_incrementally(&thresholds);
            black_box(partitions)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_partition_reconstruction_1m,
    bench_sweep_reconstruction
);
criterion_main!(benches);
