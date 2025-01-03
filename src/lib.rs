#![doc = include_str!("../README.md")]

use std::{cmp::Ordering, fmt, marker::PhantomData, num::NonZeroU32, ops};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Span {
    start: u32,
    end: u32,
}

impl Span {
    pub const UNDEFINED: Self = Self { start: 0, end: 0 };
    /// Creates a new `Span` from a range of byte indices
    ///
    /// Note: end is exclusive, it doesn't belong to the `Span`
    pub const fn new(start: u32, end: u32) -> Self {
        Span { start, end }
    }

    /// Returns a new `Span` starting at `self` and ending at `other`
    pub const fn until(&self, other: &Self) -> Self {
        Span {
            start: self.start,
            end: other.end,
        }
    }

    /// Check whether `self` was defined or is a default/unknown span
    pub fn is_defined(&self) -> bool {
        *self != Self::default()
    }

    pub fn length(&self) -> u32 {
        self.end - self.start
    }
}

impl std::ops::Index<Span> for str {
    type Output = str;

    #[inline]
    fn index(&self, span: Span) -> &str {
        &self[span.start as usize..span.end as usize]
    }
}

type Index = NonZeroU32;

/// A strongly typed reference to an arena item.
///
/// A `Handle` value can be used as an index into an [`Arena`] or [`UniqueArena`].
pub struct Handle<T> {
    index: Index,
    marker: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for Handle<T> {}

impl<T> PartialOrd for Handle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Handle<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index.cmp(&other.index)
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "[{}]", self.index)
    }
}

impl<T> Handle<T> {
    #[cfg(test)]
    pub const DUMMY: Self = Handle {
        index: unsafe { NonZeroU32::new_unchecked(u32::MAX) },
        marker: PhantomData,
    };

    pub const fn new(index: Index) -> Self {
        Handle {
            index,
            marker: PhantomData,
        }
    }

    /// Returns the zero-based index of this handle.
    pub const fn index(self) -> usize {
        let index = self.index.get() - 1;
        index as usize
    }

    /// Convert a `usize` index into a `Handle<T>`.
    fn from_usize(index: usize) -> Self {
        let handle_index = u32::try_from(index + 1)
            .ok()
            .and_then(Index::new)
            .expect("Failed to insert into arena. Handle overflows");
        Handle::new(handle_index)
    }

    /// Convert a `usize` index into a `Handle<T>`, without range checks.
    const unsafe fn from_usize_unchecked(index: usize) -> Self {
        Handle::new(Index::new_unchecked((index + 1) as u32))
    }
}

/// An arena holding a collection of reusable items of type `T`. The arena
/// supports allocating new items *contiguously*, meaning that they will
/// be stored next to each other
///
/// Adding new items to the arena produces a strongly-typed [`Handle`].
/// The arena can be indexed using the given handle to obtain
/// a reference to the stored item.
///
/// The arena *is* growable, but *not* shrinkable. Instead, elements
/// (or spans of them) will be reused after they are freed with [`Arena::free`].
#[derive(Clone)]
pub struct Arena<T> {
    /// Values of this arena, stored in a contiguous buffer. Note that when freeing elements,
    /// there will be corpses; this buffer will not be resized as long as the arena is not cleared.
    ///
    /// This itself is a contiguous buffer; the arena stores metadata in 2 separate buffers.
    data: Vec<T>,
    /// Stores alive and dead spans for elements in the area (one span can hold multiple elements).
    ///
    /// Dead spans (ones available for reuse) will also be stored in [`Arena::free_spans`].
    span_info: Vec<Span>,
    /// Stores spans that are available for reuse.
    free_spans: Vec<Span>,
}

impl<T: Clone> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug + Clone> fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<T: Clone> Arena<T> {
    /// Create a new arena with no initial capacity allocated.
    pub const fn new() -> Self {
        Arena {
            data: Vec::new(),
            span_info: Vec::new(),
            free_spans: Vec::new(),
        }
    }

    /// Create a new arena with the specified capacity allocated. Note that this does not
    /// allocate any space for span metadata.
    pub fn with_capacity(capacity: usize) -> Self {
        Arena {
            data: Vec::with_capacity(capacity),
            ..Default::default()
        }
    }

    /// Extracts the inner vector.
    #[allow(clippy::missing_const_for_fn)] // ignore due to requirement of #![feature(const_precise_live_drops)]
    pub fn into_inner(self) -> Vec<T> {
        self.data
    }

    /// Returns the current number of items stored in this arena.
    /// Note that this method is slow as it has to iterate over all spans. Use [`Arena::buffer_len`]
    /// for a faster approximation.
    pub fn len(&self) -> usize {
        self.span_info
            .iter()
            .filter(|&&span| span.is_defined())
            .map(|span| span.length() as usize)
            .sum()
    }

    /// Returns the capacity of this arena.
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Returns the length of the internal buffer of this arena. This is enough as a rough
    /// approximation, but note that it includes corpses.
    pub fn buffer_len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the arena contains no elements.
    pub fn is_empty(&self) -> bool {
        self.span_info.iter().all(|span| *span == Span::default())
    }

    /// Returns an iterator over the items stored in this arena, returning both
    /// the item's handle and a reference to it.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (Handle<T>, &T)> {
        self.data
            .iter()
            .enumerate()
            .map(|(i, v)| unsafe { (Handle::from_usize_unchecked(i), v) })
    }

    /// Drains the arena, returning an iterator over the items stored.
    pub fn drain(&mut self) -> impl DoubleEndedIterator<Item = (Handle<T>, T, Span)> {
        let arena = std::mem::take(self);
        arena
            .data
            .into_iter()
            .zip(arena.span_info)
            .enumerate()
            .map(|(i, (v, span))| unsafe { (Handle::from_usize_unchecked(i), v, span) })
    }

    /// Returns a iterator over the items stored in this arena,
    /// returning both the item's handle and a mutable reference to it.
    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (Handle<T>, &mut T)> {
        self.data
            .iter_mut()
            .enumerate()
            .map(|(i, v)| unsafe { (Handle::from_usize_unchecked(i), v) })
    }

    fn find_free_span(&mut self, count: usize) -> Option<Span> {
        if let Some(pos) = self
            .free_spans
            .iter()
            .position(|span| (span.end - span.start) as usize >= count)
        {
            let span = self.free_spans.remove(pos);
            if (span.end - span.start) as usize > count {
                // Split the span if it's larger than needed
                let new_span = Span {
                    start: span.start + count as u32,
                    end: span.end,
                };
                self.free_spans.push(new_span);
            }
            Some(Span {
                start: span.start,
                end: span.start + count as u32,
            })
        } else {
            None
        }
    }

    /// Adds a new value to the arena, returning a typed handle.
    pub fn append(&mut self, value: T, count: usize) -> Handle<T> {
        assert!(count > 0);
        if let Some(free_span) = self.find_free_span(count) {
            let start = free_span.start as usize;
            let end = start + count;
            for i in start..end - 1 {
                self.data[i] = value.clone();
            }
            self.data[end - 1] = value;
            self.span_info.push(free_span);
            return Handle::from_usize(free_span.start as usize);
        }

        let start = self.span_info.last().map_or(0, |span| span.end as usize);
        // Copy over some data into existing arena space, if any
        // | span | freespan | span | remaining len | cap |
        //                            ^^^^^^^^^^^^^ we will overwrite some of this
        if start < self.data.len() {
            for i in start..self.data.len().min(start + count) {
                self.data[i] = value.clone();
            }
        }
        // Resize the arena if needed:
        // | span | remaining len(20) | cap |
        //          ^ overwritten ^^^
        // appending 21 elements will have the last element going to the cap
        if start + count > self.data.len() {
            self.data.resize(start + count, value);
        }

        let span = Span::new(start as u32, (start + count) as u32);
        self.span_info.push(span);
        Handle::from_usize(start)
    }

    /// Fetch a handle to an existing type.
    pub fn fetch_if<F: Fn(&T) -> bool>(&self, fun: F) -> Option<Handle<T>> {
        self.data
            .iter()
            .position(fun)
            .map(|index| unsafe { Handle::from_usize_unchecked(index) })
    }

    /// Adds a value with a custom check for uniqueness:
    /// returns a handle pointing to
    /// an existing element if the check succeeds, or adds a new
    /// element otherwise.
    pub fn fetch_if_or_append<F: Fn(&T, &T) -> bool>(
        &mut self,
        value: T,
        count: usize,
        fun: F,
    ) -> Handle<T> {
        if let Some(index) = self.data.iter().position(|d| fun(d, &value)) {
            unsafe { Handle::from_usize_unchecked(index) }
        } else {
            self.append(value, count)
        }
    }

    /// Adds a value with a check for uniqueness, where the check is plain comparison.
    pub fn fetch_or_append(&mut self, value: T, count: usize) -> Handle<T>
    where
        T: PartialEq,
    {
        self.fetch_if_or_append(value, count, T::eq)
    }

    pub fn try_get(&self, handle: Handle<T>) -> Result<&T, &'static str> {
        self.data.get(handle.index()).ok_or("Handle out of range")
    }

    /// Get a mutable reference to an element in the arena.
    pub fn get_mut(&mut self, handle: Handle<T>) -> &mut T {
        self.data.get_mut(handle.index()).unwrap()
    }

    /// Clears the arena keeping all allocations
    pub fn clear(&mut self) {
        self.data.clear();
        self.span_info.clear();
        self.free_spans.clear();
    }

    pub fn get_span(&self, handle: Handle<T>) -> Span {
        self.span_info
            .iter()
            .find(|&&span| span.start == handle.index() as u32)
            .cloned()
            .unwrap()
    }

    /// Assert that `handle` is valid for this arena.
    pub fn check_contains_handle(&self, handle: Handle<T>) -> Result<(), &'static str> {
        if handle.index() < self.data.len() {
            Ok(())
        } else {
            Err("Handle out of range")
        }
    }

    pub fn retain_mut<P>(&mut self, mut predicate: P)
    where
        P: FnMut(Handle<T>, &mut T) -> bool,
    {
        let mut retained = 0;
        for i in 0..self.span_info.len() {
            let span = self.span_info[i];
            let start = span.start as usize;
            let end = span.end as usize;
            let mut keep = false;

            for j in start..end {
                let handle = Handle::from_usize(j);
                if predicate(handle, &mut self.data[j]) {
                    keep = true;
                } else {
                    self.free_spans.push(Span::new(j as u32, (j + 1) as u32));
                }
            }

            if keep {
                if retained != i {
                    self.span_info[retained] = self.span_info[i];
                    for j in start..end {
                        self.data.swap(retained * (end - start) + (j - start), j);
                    }
                }
                retained += 1;
            }
        }

        self.span_info.truncate(retained);

        // Sort and merge adjacent free spans
        self.free_spans.sort_by(|a, b| a.start.cmp(&b.start));
        let mut new_free_spans: Vec<Span> = Vec::new();
        for span in &self.free_spans {
            if let Some(last) = new_free_spans.last_mut() {
                if last.end == span.start {
                    last.end = span.end;
                } else {
                    new_free_spans.push(*span);
                }
            } else {
                new_free_spans.push(*span);
            }
        }
        self.free_spans = new_free_spans;
    }

    /// Marks the elements span as available for reuse. The new free span
    /// will be merged with existing free spans if possible. The new free
    /// spans will be sorted to make future merges easier.
    pub fn free(&mut self, handle: Handle<T>) {
        let span = self.get_span(handle);
        self.span_info.retain(|&s| s != span);

        // Attempt to merge with adjacent free spans
        let mut merged = false;
        for free_span in &mut self.free_spans {
            if free_span.end == span.start {
                free_span.end = span.end;
                merged = true;
                break;
            } else if free_span.start == span.end {
                free_span.start = span.start;
                merged = true;
                break;
            }
        }

        if !merged {
            self.free_spans.push(span);
        }

        // Sort free spans to make future merges easier
        self.free_spans.sort_by(|a, b| a.start.cmp(&b.start));

        // A free span that ends at the end of the arena is meaningless
        if let Some(last) = self.free_spans.last() {
            if last.end == self.data.len() as u32 {
                self.free_spans.pop();
            }
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    pub fn spans(&self) -> &[Span] {
        &self.span_info
    }

    #[inline]
    pub fn free_spans(&self) -> &[Span] {
        &self.free_spans
    }
}

impl<T> ops::Index<Handle<T>> for Arena<T> {
    type Output = T;
    fn index(&self, handle: Handle<T>) -> &T {
        &self.data[handle.index()]
    }
}

impl<T> ops::IndexMut<Handle<T>> for Arena<T> {
    fn index_mut(&mut self, handle: Handle<T>) -> &mut T {
        &mut self.data[handle.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    #[derive(Default)]
    struct SplitMix64 {
        x: u64,
    }

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.x = self.x.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn append_non_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.append(0, 1);
        let t2 = arena.append(0, 1);
        assert!(t1 != t2);
        assert!(arena[t1] == arena[t2]);
    }

    #[test]
    fn append_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.append(0, 1);
        let t2 = arena.append(1, 1);
        assert!(t1 != t2);
        assert!(arena[t1] != arena[t2]);
    }

    #[test]
    fn fetch_or_append_non_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.fetch_or_append(0, 1);
        let t2 = arena.fetch_or_append(0, 1);
        assert!(t1 == t2);
        assert!(arena[t1] == arena[t2])
    }

    #[test]
    fn fetch_or_append_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.fetch_or_append(0, 1);
        let t2 = arena.fetch_or_append(1, 1);
        assert!(t1 != t2);
        assert!(arena[t1] != arena[t2]);
    }

    #[test]
    fn reuse_spans() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.append(5, 10); // append 10 elements that are 5
        arena.free(t1); // kill the span
        assert!(arena.is_empty()); // arena is empty but buffer still holds 10 corpses

        let t2 = arena.append(52, 10); // again, push 10 elements, this time they are 52
        assert_eq!(t1, t2); // should start from the same place in the buffer

        // the corpses should be overwritten
        assert_eq!(arena[t2], 52);
        for i in 1..10 {
            assert_eq!(
                arena[Handle::new(NonZeroU32::new(t1.index() as u32 + i).unwrap())],
                52
            );
        }

        let t3 = arena.append(7, 42); // push 10 elements, this time they are 42
        assert_ne!(t2, t3); // should start from a new place in the buffer

        // current arena: | t2 | t3 | ... |
        arena.free(t2);
        // current arena: | corpse(10) | t3 | ... |
        let t4 = arena.append(255, 20); // 20 elements, shouldn't fit in t2's hole
        assert_ne!(t4, t3);
        assert_eq!(arena[t4], 255);
        // current arena: | corpse(10) | t3 | t4 | ... |
        arena.free(t4);
        // current arena: | corpse(10) | t3 | corpse(20) | ... |
        let t5 = arena.append(123, 21); // fits in t4's hole and takes 1 from cap
        assert_ne!(t5, t3);
        assert_eq!(arena[t5], 123);
        // current arena: | corpse(10) | t3 | t5 | ... |
        arena.free(t3);
        assert_eq!(arena.free_spans.len(), 1); // should merge span
        assert_eq!(arena.free_spans.first(), Some(&Span { start: 0, end: 52 }));
        // current arena: | corpse(52) | t5 | ... |
    }

    #[test]
    fn fuzz() {
        const NUM_ITERATIONS: usize = 1000;

        let mut arena: Arena<u64> = Arena::new();
        let mut rng = SplitMix64::default();

        let mut prev_handle = None;
        let mut prev_v = None;
        for j in 0..NUM_ITERATIONS {
            let cnt = rng.next_u64() as usize % 10 + 1;
            let v = rng.next_u64();
            let t = arena.append(v, cnt);

            // get values back
            assert_eq!(arena[t], v);
            for i in 1..cnt as u32 {
                assert_eq!(
                    arena[Handle::new(NonZeroU32::new(t.index() as u32 + i).unwrap())],
                    v
                );
            }

            // check the previous value
            if j.saturating_sub(1) % 100 != 0 {
                if let Some(prev_handle) = prev_handle {
                    assert_eq!(arena[prev_handle], prev_v.unwrap());
                }
            }

            if j % 100 == 0 {
                // free the span
                arena.free(t);
                // corpses still there
                assert_eq!(arena[t], v);
            }

            prev_handle = Some(t);
            prev_v = Some(v);
        }
    }

    #[test]
    fn arena_retain_mut() {
        let mut arena: Arena<i32> = Arena::new();
        let t1 = arena.append(5, 10);
        let t2 = arena.append(52, 10);
        let _t3 = arena.append(7, 10);
        let _t4 = arena.append(123, 21);
        let _t5 = arena.append(-7952812, 300);

        arena.retain_mut(|h, v| {
            if *v == 7 || h == t2 {
                *v = 7 * 7;
                true
            } else {
                false
            }
        });

        assert_eq!(arena[t1], 7 * 7);
        assert_eq!(arena[t2], 7 * 7);
        assert_eq!(arena.len(), 10 + 10);
    }
}
