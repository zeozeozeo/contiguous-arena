# contiguous-arena

`contiguous-arena` is a Rust crate that provides a memory-efficient, contiguous data structure for managing collections of homogeneously-typed elements. It is designed to handle frequent allocations and deallocations efficiently by reusing freed memory spans.

The library is designed to allow for allocation of contiguous elements, such that:

1. Appending an arbitrary amount of elements will allocate them in a contiguous memory block (`arena.append(T, count)`)
2. The arena uses `Vec<T>` as the backing buffer, unlike most arena crates which bundle metadata together with the item. This crate uses 2 additional buffers to store the span metadata, which makes it easier to iterate the arena with crates like `rayon` or share it with the GPU, for example.

## Features

- Memory-efficient storage by reusing freed spans.
- Contiguous memory layout for improved cache performance, easier iteration and GPU memory sharing.
- Support for arbitrary owned types. Note that the arena DOES NOT currently run `drop()`! (a PR would be appreciated)


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

## License

The Unlicense (public domain)
