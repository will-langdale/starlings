use criterion::{black_box, criterion_group, criterion_main, Criterion};
use starlings_core::core::{DataContext, Key};

fn benchmark_10m_record_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("record_insertion_production");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("1M unique record insertions", |b| {
        b.iter(|| {
            let ctx = DataContext::new();
            for i in 0..1_000_000 {
                // Mix different key types for realism
                match i % 4 {
                    0 => ctx
                        .ensure_record("customers", Key::String(format!("cust_{}", black_box(i)))),
                    1 => ctx.ensure_record("transactions", Key::U64(black_box(i as u64))),
                    2 => ctx.ensure_record("products", Key::U32(black_box(i as u32))),
                    3 => ctx
                        .ensure_record("addresses", Key::Bytes(format!("addr_{}", i).into_bytes())),
                    _ => unreachable!(),
                };
            }
            black_box(ctx.len())
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_10m_record_insertion);
criterion_main!(benches);
