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

criterion_group!(benches, bench_append, bench_free, bench_reuse_spans);
criterion_main!(benches);
