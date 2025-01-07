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

```
append 1 element        time:   [182.92 ns 191.00 ns 199.09 ns]
append 10 elements      time:   [166.67 ns 170.09 ns 174.47 ns]
append 100 elements     time:   [170.19 ns 174.47 ns 179.53 ns]
free 10 elements        time:   [248.43 ns 254.22 ns 260.94 ns]
free 100 elements       time:   [248.16 ns 251.57 ns 255.99 ns]
reuse span of 10 elements
                        time:   [245.90 ns 247.98 ns 250.80 ns]
reuse span of 100 elements
                        time:   [311.23 ns 323.52 ns 336.90 ns]        
```

(light travels about 30 meters in 100 nanoseconds)

## License

The Unlicense (public domain)
