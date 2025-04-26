# contiguous-arena

`contiguous-arena` is a Rust crate that provides a memory-efficient, contiguous data structure for managing collections of homogeneously-typed elements. It is designed to handle frequent allocations and deallocations efficiently by reusing freed memory spans.

The library is designed to allow for allocation of contiguous elements, such that:

1. Appending an arbitrary amount of elements will allocate them in a contiguous memory block (`arena.append(T, count)`)
2. The arena uses `Vec<T>` as the backing buffer, unlike most arena crates which bundle metadata together with the item. This crate uses 2 additional buffers to store the span metadata, which makes it easier to iterate the arena with crates like `rayon` or share it with the GPU, for example.

## Features

- Memory-efficient storage by reusing freed spans.
- Contiguous memory layout for improved cache performance, easier iteration and GPU memory sharing.
- Support for arbitrary owned types. Note that the arena DOES NOT currently run `drop()`! (a PR would be appreciated)
- Only depends on std. A PR adding `#![no_std]` support would be appreciated.

Be aware of memory fragmentation when using this library: frequest allocations of different sizes will eventually fragment the arena. This crate is ideal for use-cases where most allocations are about the same size.

## Example

```rust
use contiguous_arena::Arena;

fn main() {
    let mut arena: Arena<u8> = Arena::new();

    // Append elements to the arena
    let handle1 = arena.append(5, 10); // Append 10 elements of value 5
    let handle2 = arena.append(6, 10); // Append 10 elements of value 6

    // Access elements using handles
    assert_eq!(arena[handle1], 5);
    assert_eq!(arena[handle2], 6);

    // Free a span
    arena.free(handle1);

    // Reuse freed span
    let handle3 = arena.append(7, 10); // This should reuse the space freed by handle1
    assert_eq!(handle1, handle3);
    assert_eq!(arena[handle3], 7);
}
```

## Benchmarks and tests

To run tests, run `cargo test`. To run benchmarks, run `cargo bench` in the project directory. See benchmark results in `target/criterion/report/index.html`.

Benchmark results on an Intel Xeon E5-2690 v3 with 2133 MHz DDR4 quad-channel RAM:

```ignore
append 1 element        time:   [122.48 ns 125.71 ns 129.62 ns]
append 10 elements      time:   [126.89 ns 129.69 ns 133.18 ns]
append 100 elements     time:   [129.01 ns 129.65 ns 130.51 ns]
free 10 elements        time:   [185.96 ns 189.48 ns 193.88 ns]
free 100 elements       time:   [191.01 ns 191.40 ns 191.93 ns]
reuse span of 10 elements
                        time:   [205.49 ns 210.46 ns 216.89 ns]
reuse span of 100 elements
                        time:   [266.60 ns 274.03 ns 282.45 ns]
iterate 100 elements    time:   [85.580 ns 87.331 ns 89.547 ns]
iterate 1000 elements   time:   [696.59 ns 715.24 ns 736.26 ns]
iterate and mutate 100 elements
                        time:   [67.295 ns 67.438 ns 67.636 ns]
iterate and mutate 1000 elements
                        time:   [625.11 ns 639.24 ns 656.52 ns]
fetch_if found in 100 elements
                        time:   [66.565 ns 68.174 ns 70.049 ns]
fetch_if not found in 1000 elements
                        time:   [582.72 ns 591.28 ns 603.65 ns]
fetch_or_append found in 100 elements
                        time:   [2.5913 ns 2.5918 ns 2.5922 ns]
fetch_or_append not found in 1000 elements
                        time:   [585.43 ns 591.74 ns 600.30 ns]
retain_mut keep half of 100 elements
                        time:   [325.25 ns 333.94 ns 345.95 ns]
mixed workload with 100 elements
                        time:   [433.85 ns 439.76 ns 452.24 ns]
```

(light travels about 30 meters in 100 nanoseconds)

## License

The Unlicense (public domain)
