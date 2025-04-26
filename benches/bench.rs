use contiguous_arena::Arena;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_append(c: &mut Criterion) {
    c.bench_function("append 1 element", |b| {
        b.iter(|| {
            let mut arena: Arena<u8> = Arena::new();
            arena.append(black_box(5), black_box(1));
        })
    });

    c.bench_function("append 10 elements", |b| {
        b.iter(|| {
            let mut arena: Arena<u8> = Arena::new();
            arena.append(black_box(5), black_box(10));
        })
    });

    c.bench_function("append 100 elements", |b| {
        b.iter(|| {
            let mut arena: Arena<u8> = Arena::new();
            arena.append(black_box(5), black_box(100));
        })
    });
}

fn bench_free(c: &mut Criterion) {
    c.bench_function("free 10 elements", |b| {
        b.iter(|| {
            let mut arena: Arena<u8> = Arena::new();
            let handle = arena.append(black_box(5), black_box(10));
            arena.free(handle);
        })
    });

    c.bench_function("free 100 elements", |b| {
        b.iter(|| {
            let mut arena: Arena<u8> = Arena::new();
            let handle = arena.append(black_box(5), black_box(100));
            arena.free(handle);
        })
    });
}

fn bench_reuse_spans(c: &mut Criterion) {
    c.bench_function("reuse span of 10 elements", |b| {
        b.iter(|| {
            let mut arena: Arena<u8> = Arena::new();
            let handle1 = arena.append(black_box(5), black_box(10));
            arena.free(handle1);
            let handle2 = arena.append(black_box(6), black_box(10));
            black_box(handle2);
        })
    });

    c.bench_function("reuse span of 100 elements", |b| {
        b.iter(|| {
            let mut arena: Arena<u8> = Arena::new();
            let handle1 = arena.append(black_box(5), black_box(100));
            arena.free(handle1);
            let handle2 = arena.append(black_box(6), black_box(100));
            black_box(handle2);
        })
    });
}

fn bench_iter(c: &mut Criterion) {
    c.bench_function("iterate 100 elements", |b| {
        let mut arena: Arena<u8> = Arena::new();
        arena.append(black_box(5), black_box(100));
        b.iter(|| {
            for (_, value) in arena.iter() {
                black_box(value);
            }
        })
    });

    c.bench_function("iterate 1000 elements", |b| {
        let mut arena: Arena<u8> = Arena::new();
        arena.append(black_box(5), black_box(1000));
        b.iter(|| {
            for (_, value) in arena.iter() {
                black_box(value);
            }
        })
    });
}

fn bench_iter_mut(c: &mut Criterion) {
    c.bench_function("iterate and mutate 100 elements", |b| {
        let mut arena: Arena<u8> = Arena::new();
        arena.append(black_box(5), black_box(100));
        b.iter(|| {
            for (_, value) in arena.iter_mut() {
                *value = black_box(10);
            }
        })
    });

    c.bench_function("iterate and mutate 1000 elements", |b| {
        let mut arena: Arena<u8> = Arena::new();
        arena.append(black_box(5), black_box(1000));
        b.iter(|| {
            for (_, value) in arena.iter_mut() {
                *value = black_box(10);
            }
        })
    });
}

fn bench_fetch_if(c: &mut Criterion) {
    c.bench_function("fetch_if found in 100 elements", |b| {
        let mut arena: Arena<u8> = Arena::new();
        arena.append(black_box(5), black_box(99));
        arena.append(black_box(6), black_box(1));
        b.iter(|| {
            let handle = arena.fetch_if(|&v| v == black_box(6));
            black_box(handle);
        })
    });

    c.bench_function("fetch_if not found in 1000 elements", |b| {
        let mut arena: Arena<u8> = Arena::new();
        arena.append(black_box(5), black_box(1000));
        b.iter(|| {
            let handle = arena.fetch_if(|&v| v == black_box(6));
            black_box(handle);
        })
    });
}

fn bench_fetch_or_append(c: &mut Criterion) {
    c.bench_function("fetch_or_append found in 100 elements", |b| {
        let mut arena: Arena<u8> = Arena::new();
        arena.append(black_box(5), black_box(100));
        b.iter(|| {
            let handle = arena.fetch_or_append(black_box(5), black_box(1));
            black_box(handle);
        })
    });

    c.bench_function("fetch_or_append not found in 1000 elements", |b| {
        let mut arena: Arena<u8> = Arena::new();
        arena.append(black_box(5), black_box(1000));
        b.iter(|| {
            let handle = arena.fetch_or_append(black_box(6), black_box(1));
            black_box(handle);
        })
    });
}

fn bench_retain_mut(c: &mut Criterion) {
    c.bench_function("retain_mut keep half of 100 elements", |b| {
        let mut arena: Arena<u8> = Arena::new();
        arena.append(black_box(5), black_box(50));
        arena.append(black_box(6), black_box(50));
        b.iter(|| {
            let mut arena_clone = arena.clone();
            arena_clone.retain_mut(|_, v| *v == black_box(5));
            black_box(&arena_clone);
        })
    });

    c.bench_function("retain_mut keep none of 1000 elements", |b| {
        let mut arena: Arena<u8> = Arena::new();
        arena.append(black_box(5), black_box(1000));
        b.iter(|| {
            let mut arena_clone = arena.clone();
            arena_clone.retain_mut(|_, v| *v == black_box(6));
            black_box(&arena_clone);
        })
    });
}

fn bench_mixed_workload(c: &mut Criterion) {
    c.bench_function("mixed workload with 100 elements", |b| {
        b.iter(|| {
            let mut arena: Arena<u8> = Arena::new();
            let h1 = arena.append(black_box(1), black_box(50));
            let h2 = arena.append(black_box(2), black_box(50));
            for (_, value) in arena.iter_mut() {
                *value = black_box(*value + 1);
            }
            arena.free(h1);
            let h3 = arena.append(black_box(3), black_box(25));
            let handle = arena.fetch_if(|&v| v == black_box(3));
            black_box((h2, h3, handle));
        })
    });
}

criterion_group!(
    benches,
    bench_append,
    bench_free,
    bench_reuse_spans,
    bench_iter,
    bench_iter_mut,
    bench_fetch_if,
    bench_fetch_or_append,
    bench_retain_mut,
    bench_mixed_workload
);
criterion_main!(benches);
