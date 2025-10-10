use criterion::{black_box, criterion_group, criterion_main, Criterion};
use starlings_core::core::{DataContext, Key};
use starlings_core::hierarchy::PartitionHierarchy;
use starlings_core::test_utils::generate_entity_resolution_edges;
use std::sync::Arc;

fn generate_test_edges(entity_count: usize) -> (Vec<(u32, u32, f64)>, Arc<DataContext>) {
    let ctx = DataContext::new();

    // Create mixed record types for realistic benchmarking
    for i in 0..entity_count {
        match i % 4 {
            0 => ctx.ensure_record("customers", Key::U32(i as u32)),
            1 => ctx.ensure_record("transactions", Key::U64(1000000 + i as u64)),
            2 => ctx.ensure_record("products", Key::U32((i + 1000000) as u32)),
            3 => ctx.ensure_record("addresses", Key::U32(i as u32)),
            _ => unreachable!(),
        };
    }

    let ctx = Arc::new(ctx);

    // Use unified entity resolution generator for realistic patterns
    let edges = generate_entity_resolution_edges(entity_count, Some(10)); // Discrete thresholds for consistent timing

    (edges, ctx)
}

fn bench_hierarchy_construction_progressive(c: &mut Criterion) {
    // Entity counts that produce realistic edge counts (~entity_count * 5 edges after deduplication)
    let scales = vec![
        ("20k_entities", 20_000),
        ("100k_entities", 100_000),
        ("200k_entities", 200_000),
    ];

    let mut group = c.benchmark_group("hierarchy_construction");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    for (name, entity_count) in scales {
        let (edges, ctx) = generate_test_edges(entity_count);

        group.bench_function(format!("{}_{}_edges", name, edges.len()), |b| {
            b.iter(|| {
                black_box(
                    PartitionHierarchy::from_edges(edges.clone(), ctx.clone(), 6, None).unwrap(),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hierarchy_construction_progressive);
criterion_main!(benches);
