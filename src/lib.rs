#![doc = include_str!("../README.md")]

use std::{cmp::Ordering, fmt, marker::PhantomData, num::NonZeroU32, ops};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Span {
    start: u32,
    end: u32,
}

impl Span {
    pub const UNDEFINED: Self = Self { start: 0, end: 0 };

    pub const fn new(start: u32, end: u32) -> Self {
        debug_assert!(start <= end, "Span start must be <= end");
        Span { start, end }
    }

    pub fn is_defined(&self) -> bool {
        self.start != self.end
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
        index: NonZeroU32::new(u32::MAX).unwrap(),
        marker: PhantomData,
    };

    #[inline]
    pub const fn new(index: Index) -> Self {
        Handle {
            index,
            marker: PhantomData,
        }
    }

    #[inline]
    pub const fn index(self) -> usize {
        let index = self.index.get() - 1;
        index as usize
    }

    #[inline]
    fn from_usize(index: usize) -> Self {
        let handle_index = u32::try_from(index + 1)
            .ok()
            .and_then(Index::new)
            .expect("failed to insert into arena");
        Handle::new(handle_index)
    }

    #[inline]
    const unsafe fn from_usize_unchecked(index: usize) -> Self {
        Handle::new(Index::new_unchecked((index + 1) as u32))
    }
}

#[derive(Clone)]
pub struct Arena<T> {
    data: Vec<T>,
    span_info: Vec<Span>,  // Kept sorted by span.start
    free_spans: Vec<Span>, // Kept sorted by span.start
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
    pub const fn new() -> Self {
        Arena {
            data: Vec::new(),
            span_info: Vec::new(),
            free_spans: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Arena {
            data: Vec::with_capacity(capacity),
            span_info: Vec::new(),
            free_spans: Vec::new(),
        }
    }

    pub fn into_inner(self) -> Vec<T> {
        self.data
    }

    pub fn len(&self) -> usize {
        self.span_info
            .iter()
            .map(|span| span.length() as usize)
            .sum()
    }

    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    pub fn buffer_len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.span_info.is_empty()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (Handle<T>, &T)> + '_ {
        self.span_info
            .iter()
            .flat_map(move |&span| (span.start..span.end))
            .map(move |i| {
                let handle = unsafe { Handle::from_usize_unchecked(i as usize) };
                (handle, &self.data[i as usize])
            })
    }

    pub fn drain(&mut self) -> impl Iterator<Item = (Handle<T>, T, Span)> + '_ {
        let arena = std::mem::take(self);
        arena
            .data
            .into_iter()
            .zip(
                arena
                    .span_info
                    .into_iter()
                    .flat_map(|span| std::iter::repeat_n(span, span.length() as usize)),
            )
            .enumerate()
            .map(|(i, (v, span))| unsafe { (Handle::from_usize_unchecked(i), v, span) })
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (Handle<T>, &mut T)> + '_ {
        let data_ptr = self.data.as_mut_ptr();
        self.span_info
            .iter()
            .flat_map(move |&span| (span.start..span.end))
            .map(move |i| {
                let handle = unsafe { Handle::from_usize_unchecked(i as usize) };
                let value_ref = unsafe { &mut *data_ptr.add(i as usize) };
                (handle, value_ref)
            })
    }

    fn find_free_span(&mut self, count: u32) -> Option<Span> {
        if let Some(pos) = self
            .free_spans
            .iter()
            .position(|span| span.length() >= count)
        {
            let span = self.free_spans.remove(pos);

            if span.length() > count {
                let remaining_span = Span {
                    start: span.start + count,
                    end: span.end,
                };
                match self.free_spans.binary_search(&remaining_span) {
                    Ok(idx) => self.free_spans.insert(idx, remaining_span),
                    Err(idx) => self.free_spans.insert(idx, remaining_span),
                }
            }
            Some(Span {
                start: span.start,
                end: span.start + count,
            })
        } else {
            None
        }
    }

    pub fn append(&mut self, value: T, count: usize) -> Handle<T> {
        assert!(count > 0);
        let u_count = count as u32;

        if let Some(target_span) = self.find_free_span(u_count) {
            let start = target_span.start as usize;
            let end = target_span.end as usize;
            for i in start..end - 1 {
                self.data[i] = value.clone();
            }
            self.data[end - 1] = value;

            match self
                .span_info
                .binary_search_by_key(&target_span.start, |s| s.start)
            {
                Ok(_) => unreachable!("should not find existing span at reused start"),
                Err(idx) => self.span_info.insert(idx, target_span),
            }

            return Handle::from_usize(start);
        }

        let start_idx = self.data.len();
        let end_idx = start_idx + count;

        self.data.resize(end_idx, value);

        let new_span = Span::new(start_idx as u32, end_idx as u32);
        self.span_info.push(new_span);

        Handle::from_usize(start_idx)
    }

    pub fn fetch_if<F: Fn(&T) -> bool>(&self, fun: F) -> Option<Handle<T>> {
        self.iter()
            .find(|(_, data)| fun(data))
            .map(|(handle, _)| handle)
    }

    pub fn fetch_if_or_append<F: Fn(&T, &T) -> bool>(
        &mut self,
        value: T,
        count: usize,
        fun: F,
    ) -> Handle<T> {
        let found_handle = self
            .iter()
            .find(|(_, data)| fun(data, &value))
            .map(|(h, _)| h);

        if let Some(handle) = found_handle {
            handle
        } else {
            self.append(value, count)
        }
    }

    pub fn fetch_or_append(&mut self, value: T, count: usize) -> Handle<T>
    where
        T: PartialEq,
    {
        self.fetch_if_or_append(value, count, T::eq)
    }

    pub fn try_get(&self, handle: Handle<T>) -> Result<&T, &'static str> {
        let index = handle.index();
        if index < self.data.len()
            && self
                .span_info
                .binary_search_by(|probe| {
                    if index < probe.start as usize {
                        Ordering::Greater
                    } else if index >= probe.end as usize {
                        Ordering::Less
                    } else {
                        Ordering::Equal
                    }
                })
                .is_ok()
        {
            Ok(unsafe { self.data.get_unchecked(index) })
        } else {
            Err("Handle out of range or points to freed memory")
        }
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> &mut T {
        let index = handle.index();
        let data_len = self.data.len();
        let span_exists = self
            .span_info
            .binary_search_by(|probe| {
                if index < probe.start as usize {
                    Ordering::Greater
                } else if index >= probe.end as usize {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .is_ok();

        if index < data_len && span_exists {
            unsafe { self.data.get_unchecked_mut(index) }
        } else {
            panic!("Handle out of range or points to freed memory");
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.span_info.clear();
        self.free_spans.clear();
    }

    pub fn get_span(&self, handle: Handle<T>) -> Span {
        let index = handle.index() as u32;
        self.span_info
            .binary_search_by(|probe| {
                if index < probe.start {
                    Ordering::Greater
                } else if index >= probe.end {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .map(|idx| self.span_info[idx])
            .expect("Handle does not point to a valid span")
    }

    pub fn check_contains_handle(&self, handle: Handle<T>) -> Result<(), &'static str> {
        self.try_get(handle).map(|_| ())
    }

    pub fn retain_mut<P>(&mut self, mut predicate: P)
    where
        P: FnMut(Handle<T>, &mut T) -> bool,
    {
        let mut handles_to_free = Vec::new();
        let data_ptr = self.data.as_mut_ptr();

        for span in &self.span_info {
            let mut keep_span = false;
            for i in span.start..span.end {
                let handle = unsafe { Handle::from_usize_unchecked(i as usize) };
                let value_ref = unsafe { &mut *data_ptr.add(i as usize) };
                if predicate(handle, value_ref) {
                    keep_span = true;
                }
            }
            if !keep_span {
                handles_to_free.push(unsafe { Handle::from_usize_unchecked(span.start as usize) });
            }
        }

        for handle in handles_to_free {
            self.free(handle);
        }
    }

    pub fn free(&mut self, handle: Handle<T>) {
        let index = handle.index() as u32;
        let span_idx = match self.span_info.binary_search_by_key(&index, |s| s.start) {
            Ok(idx) if self.span_info[idx].start == index => idx,
            _ => panic!(
                "attempted to free an invalid handle (not start of a span) or already freed span"
            ),
        };

        let span_to_free = self.span_info.remove(span_idx);

        let mut merged_span = span_to_free;

        if let Ok(right_idx) = self
            .free_spans
            .binary_search_by_key(&merged_span.end, |s| s.start) {
            merged_span.end = self.free_spans.remove(right_idx).end;
        }

        let left_search = self
            .free_spans
            .binary_search_by(|probe| probe.end.cmp(&merged_span.start));
        match left_search {
            Ok(left_idx) => {
                merged_span.start = self.free_spans.remove(left_idx).start;
            }
            Err(idx) if idx > 0 => {
                if self.free_spans[idx - 1].end == merged_span.start {
                    merged_span.start = self.free_spans.remove(idx - 1).start;
                }
            }
            _ => {}
        }

        match self.free_spans.binary_search(&merged_span) {
            Ok(_) => unreachable!("Should not have duplicate free spans"),
            Err(idx) => self.free_spans.insert(idx, merged_span),
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

impl<T: Clone> ops::Index<Handle<T>> for Arena<T> {
    type Output = T;
    #[inline]
    fn index(&self, handle: Handle<T>) -> &T {
        match self.try_get(handle) {
            Ok(val) => val,
            Err(msg) => panic!("{}", msg),
        }
    }
}

impl<T: Clone> ops::IndexMut<Handle<T>> for Arena<T> {
    #[inline]
    fn index_mut(&mut self, handle: Handle<T>) -> &mut T {
        self.get_mut(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn append_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.append(0, 1);
        let t2 = arena.append(1, 1);
        assert!(t1 != t2);
        assert!(arena[t1] != arena[t2]);
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn fetch_or_append_non_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.fetch_or_append(0, 1);
        let t2 = arena.fetch_or_append(0, 1);
        assert!(t1 == t2);
        assert!(arena[t1] == arena[t2]);
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn fetch_or_append_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.fetch_or_append(0, 1);
        let t2 = arena.fetch_or_append(1, 1);
        assert!(t1 != t2);
        assert!(arena[t1] != arena[t2]);
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn iterators_live_only() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 3); // indices 0, 1, 2
        let _h2 = arena.append(2, 2); // indices 3, 4
        arena.free(h1);
        let _h3 = arena.append(3, 1); // index 5
        let _h4 = arena.append(4, 2); // indices 0, 1 (reused)

        let mut live_elements = arena.iter().map(|(_, &v)| v).collect::<Vec<_>>();
        live_elements.sort_unstable();
        assert_eq!(live_elements, vec![2, 2, 3, 4, 4]);

        assert_eq!(arena.len(), 5);

        for (_, val) in arena.iter_mut() {
            *val *= 10;
        }
        let mut live_elements_mut = arena.iter().map(|(_, &v)| v).collect::<Vec<_>>();
        live_elements_mut.sort_unstable();
        assert_eq!(live_elements_mut, vec![20, 20, 30, 40, 40]);
    }

    #[test]
    fn reuse_spans() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.append(5, 10); // Span 0..10
        assert_eq!(arena.len(), 10);
        assert_eq!(arena.spans(), &[Span::new(0, 10)]);
        arena.free(t1); // kill the span
        assert!(arena.is_empty());
        assert_eq!(arena.free_spans(), &[Span::new(0, 10)]);

        let t2 = arena.append(52, 10); // Reuse 0..10
        assert_eq!(t1, t2); // should start from the same place
        assert!(arena.free_spans().is_empty());
        assert_eq!(arena.spans(), &[Span::new(0, 10)]);
        assert_eq!(arena.len(), 10);

        assert_eq!(arena[t2], 52);
        for i in 1..10 {
            let handle = unsafe { Handle::from_usize_unchecked(t1.index() + i as usize) };
            assert_eq!(arena[handle], 52);
        }

        let t3 = arena.append(7, 42); // Append 10..52
        assert_ne!(t2, t3); // should start from a new place
        assert_eq!(arena.spans(), &[Span::new(0, 10), Span::new(10, 52)]);
        assert_eq!(arena.len(), 10 + 42);

        arena.free(t2); // Free 0..10
        assert_eq!(arena.free_spans(), &[Span::new(0, 10)]);
        assert_eq!(arena.spans(), &[Span::new(10, 52)]);
        assert_eq!(arena.len(), 42);

        let t4 = arena.append(255, 20); // Append 52..72 (doesn't fit in free span)
        assert_ne!(t4.index(), 0); // Did not reuse 0..10
        assert_eq!(t4.index(), 52);
        assert_eq!(arena.free_spans(), &[Span::new(0, 10)]);
        assert_eq!(arena.spans(), &[Span::new(10, 52), Span::new(52, 72)]);
        assert_eq!(arena.len(), 42 + 20);

        arena.free(t4); // Free 52..72
        assert_eq!(arena.free_spans(), &[Span::new(0, 10), Span::new(52, 72)]);
        assert_eq!(arena.spans(), &[Span::new(10, 52)]);
        assert_eq!(arena.len(), 42);

        let t5 = arena.append(123, 5); // Reuse part of 0..10 -> 0..5
        assert_eq!(t5.index(), 0);
        assert_eq!(arena[t5], 123);
        assert_eq!(arena.free_spans(), &[Span::new(5, 10), Span::new(52, 72)]); // Remainder 5..10 is free
        assert_eq!(arena.spans(), &[Span::new(0, 5), Span::new(10, 52)]); // t5, t3 (sorted)
        assert_eq!(arena.len(), 5 + 42);

        arena.free(t3); // Free 10..52
        assert_eq!(arena.free_spans(), &[Span::new(5, 72)]);
        assert_eq!(arena.spans(), &[Span::new(0, 5)]); // Only t5 remains
        assert_eq!(arena.len(), 5);
    }

    #[test]
    fn fuzz() {
        const NUM_ITERATIONS: usize = 1000;

        let mut arena: Arena<u64> = Arena::new();
        let mut rng = SplitMix64::default();
        let mut live_handles = Vec::new();

        for j in 0..NUM_ITERATIONS {
            let cnt = (rng.next_u64() % 10 + 1) as usize;
            let v = rng.next_u64();
            let t = arena.append(v, cnt);
            live_handles.push((t, v, cnt));

            assert_eq!(arena[t], v);
            for i in 1..cnt {
                let handle = unsafe { Handle::from_usize_unchecked(t.index() + i) };
                assert_eq!(arena[handle], v);
            }

            if j > 10 && j % 5 == 0 && !live_handles.is_empty() {
                let idx_to_check = rng.next_u64() as usize % live_handles.len();
                let (prev_handle, prev_v, prev_cnt) = live_handles[idx_to_check];
                assert_eq!(arena[prev_handle], prev_v);
                for i in 1..prev_cnt {
                    let handle = unsafe { Handle::from_usize_unchecked(prev_handle.index() + i) };
                    assert_eq!(arena[handle], prev_v);
                }
            }

            if j > 5 && j % 10 == 0 && !live_handles.is_empty() {
                let idx_to_remove = rng.next_u64() as usize % live_handles.len();
                let (handle_to_free, _, _) = live_handles.remove(idx_to_remove);
                arena.free(handle_to_free);
                assert!(arena.try_get(handle_to_free).is_err());
            }
        }

        let mut total_len = 0;
        for (handle, v, cnt) in &live_handles {
            assert_eq!(arena[*handle], *v);
            total_len += cnt;
            for i in 1..*cnt {
                let h = unsafe { Handle::from_usize_unchecked(handle.index() + i) };
                assert_eq!(arena[h], *v);
            }
        }
        assert_eq!(arena.len(), total_len);
    }

    #[test]
    fn arena_retain_mut() {
        let mut arena: Arena<i32> = Arena::new();
        let h1 = arena.append(5, 10); // Span 0..10
        let h2 = arena.append(52, 10); // Span 10..20
        let h3 = arena.append(7, 10); // Span 20..30
        let h4 = arena.append(123, 21); // Span 30..51
        let h5 = arena.append(-7952812, 300); // Span 51..351

        let initial_len = arena.len();
        assert_eq!(initial_len, 10 + 10 + 10 + 21 + 300);

        arena.retain_mut(|h, v| {
            let h_idx = h.index();
            if h_idx >= h2.index() && h_idx < h3.index() + 10 {
                if *v == 7 {
                    *v = 49;
                }
                true
            } else {
                false
            }
        });

        assert!(arena.try_get(h1).is_err());
        assert!(arena.try_get(h4).is_err());
        assert!(arena.try_get(h5).is_err());

        assert!(arena.try_get(h2).is_ok());
        assert!(arena.try_get(h3).is_ok());

        assert_eq!(arena[h2], 52);
        for i in 0..10 {
            let handle = unsafe { Handle::from_usize_unchecked(h2.index() + i) };
            assert_eq!(arena[handle], 52);
        }
        assert_eq!(arena[h3], 49);
        for i in 0..10 {
            let handle = unsafe { Handle::from_usize_unchecked(h3.index() + i) };
            assert_eq!(arena[handle], 49);
        }

        assert_eq!(arena.len(), 10 + 10);

        assert_eq!(arena.free_spans(), &[Span::new(0, 10), Span::new(30, 351)]);
    }

    #[test]
    fn get_span_test() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 5); // 0..5
        let h2 = arena.append(2, 10); // 5..15
        arena.free(h1);
        let h3 = arena.append(3, 3); // 0..3
        let h4 = arena.append(4, 4); // 15..19

        let handle_in_h3 = unsafe { Handle::from_usize_unchecked(h3.index() + 1) }; // 1 in 0..3
        let handle_in_h2 = unsafe { Handle::from_usize_unchecked(h2.index() + 5) }; // 10 in 5..15
        let handle_in_h4 = unsafe { Handle::from_usize_unchecked(h4.index() + 2) }; // 17 in 15..19

        assert_eq!(arena.get_span(handle_in_h3), Span::new(0, 3));
        assert_eq!(arena.get_span(handle_in_h2), Span::new(5, 15));
        assert_eq!(arena.get_span(handle_in_h4), Span::new(15, 19));
    }

    #[test]
    #[should_panic]
    fn get_span_panic_freed() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 5);
        arena.free(h1);
        arena.get_span(h1);
    }

    #[test]
    #[should_panic]
    fn index_panic_freed() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 5);
        arena.free(h1);
        let _ = arena[h1];
    }

    #[test]
    #[should_panic]
    fn index_mut_panic_freed() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 5);
        arena.free(h1);
        arena[h1] = 9;
    }

    #[test]
    #[should_panic]
    fn free_panic_middle_of_span() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 5); // Span 0..5
        let handle_in_middle = unsafe { Handle::from_usize_unchecked(h1.index() + 2) }; // Index 2
        arena.free(handle_in_middle);
    }

    #[test]
    fn append_after_free_all() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 5); // Span 0..5
        let h2 = arena.append(2, 5); // Span 5..10
        arena.free(h1);
        arena.free(h2);
        assert!(arena.is_empty());
        assert_eq!(arena.free_spans(), &[Span::new(0, 10)]);

        let h3 = arena.append(3, 10); // Should reuse 0..10
        assert_eq!(h3.index(), 0);
        assert_eq!(arena[h3], 3);
        for i in 0..10 {
            let handle = unsafe { Handle::from_usize_unchecked(i) };
            assert_eq!(arena[handle], 3);
        }
        assert_eq!(arena.spans(), &[Span::new(0, 10)]);
        assert!(arena.free_spans().is_empty());
    }

    #[test]
    fn merge_free_spans() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 5); // 0..5
        let h2 = arena.append(2, 5); // 5..10
        let h3 = arena.append(3, 5); // 10..15
        arena.free(h1); // Free 0..5
        arena.free(h3); // Free 10..15
        assert_eq!(arena.free_spans(), &[Span::new(0, 5), Span::new(10, 15)]);

        arena.free(h2); // Free 5..10, should merge with 0..5 and 10..15
        assert_eq!(arena.free_spans(), &[Span::new(0, 15)]);
        assert!(arena.spans().is_empty());
    }

    #[test]
    fn append_exact_fit() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 10); // 0..10
        arena.free(h1); // Free 0..10
        let h2 = arena.append(2, 10); // Should reuse 0..10 exactly
        assert_eq!(h2.index(), 0);
        assert_eq!(arena[h2], 2);
        assert_eq!(arena.spans(), &[Span::new(0, 10)]);
        assert!(arena.free_spans().is_empty());
    }

    #[test]
    fn append_larger_than_free() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 5); // 0..5
        arena.free(h1); // Free 0..5
        let h2 = arena.append(2, 10); // Needs 10, only 5 free, so append at 5..15
        assert_eq!(h2.index(), 5);
        assert_eq!(arena[h2], 2);
        assert_eq!(arena.spans(), &[Span::new(5, 15)]);
        assert_eq!(arena.free_spans(), &[Span::new(0, 5)]);
    }

    #[test]
    fn multiple_handles_same_span() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 5); // 0..5
        for i in 0..5 {
            let handle = unsafe { Handle::from_usize_unchecked(h1.index() + i) };
            assert_eq!(arena.get_span(handle), Span::new(0, 5));
        }
    }

    #[test]
    fn iterate_arena() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 3); // 0..3
        let h2 = arena.append(2, 2); // 3..5
        let mut iter = arena.iter();
        assert_eq!(iter.next(), Some((h1, &1)));
        assert_eq!(
            iter.next(),
            Some((unsafe { Handle::from_usize_unchecked(1) }, &1))
        );
        assert_eq!(
            iter.next(),
            Some((unsafe { Handle::from_usize_unchecked(2) }, &1))
        );
        assert_eq!(iter.next(), Some((h2, &2)));
        assert_eq!(
            iter.next(),
            Some((unsafe { Handle::from_usize_unchecked(4) }, &2))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn drain_arena() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 3); // 0..3
        let h2 = arena.append(2, 2); // 3..5
        let drained: Vec<_> = arena.drain().collect();
        assert_eq!(drained.len(), 5);
        assert_eq!(drained[0], (h1, 1, Span::new(0, 3)));
        assert_eq!(
            drained[1],
            (
                unsafe { Handle::from_usize_unchecked(1) },
                1,
                Span::new(0, 3)
            )
        );
        assert_eq!(
            drained[2],
            (
                unsafe { Handle::from_usize_unchecked(2) },
                1,
                Span::new(0, 3)
            )
        );
        assert_eq!(drained[3], (h2, 2, Span::new(3, 5)));
        assert_eq!(
            drained[4],
            (
                unsafe { Handle::from_usize_unchecked(4) },
                2,
                Span::new(3, 5)
            )
        );
        assert!(arena.is_empty());
        assert!(arena.spans().is_empty());
        assert!(arena.free_spans().is_empty());
    }

    #[test]
    fn test_with_struct() {
        #[derive(Debug, Clone, PartialEq)]
        struct TestStruct {
            a: u32,
            b: u32,
        }

        let mut arena: Arena<TestStruct> = Arena::new();
        let val1 = TestStruct { a: 1, b: 2 };
        let val2 = TestStruct { a: 3, b: 4 };
        let h1 = arena.append(val1.clone(), 5);
        let h2 = arena.append(val2.clone(), 5);
        assert_eq!(arena[h1], val1);
        assert_eq!(arena[h2], val2);
        arena.free(h1);
        let h3 = arena.append(val2.clone(), 5);
        assert_eq!(h3.index(), 0);
        assert_eq!(arena[h3], val2);
    }

    #[test]
    fn test_zero_sized() {
        #[derive(Clone, Debug, PartialEq)]
        struct ZeroSized;

        let mut arena: Arena<ZeroSized> = Arena::new();
        let h1 = arena.append(ZeroSized, 10);
        assert_eq!(arena.len(), 10);
        for i in 0..10 {
            let handle = unsafe { Handle::from_usize_unchecked(h1.index() + i) };
            assert_eq!(arena[handle], ZeroSized);
        }
        arena.free(h1);
        assert!(arena.is_empty());
    }

    #[test]
    fn test_fetch_if() {
        let mut arena: Arena<u8> = Arena::new();
        let _h1 = arena.append(1, 5);
        let h2 = arena.append(2, 5);
        let found = arena.fetch_if(|&v| v == 2);
        assert_eq!(found, Some(h2));
        let not_found = arena.fetch_if(|&v| v == 3);
        assert_eq!(not_found, None);
    }

    #[test]
    fn test_fetch_if_or_append_custom() {
        #[derive(Clone, Debug)]
        struct Person {
            name: String,
            age: u32,
        }

        let mut arena: Arena<Person> = Arena::new();
        let alice1 = Person {
            name: "Alice".to_string(),
            age: 30,
        };
        let alice2 = Person {
            name: "Alice".to_string(),
            age: 31,
        };
        let bob = Person {
            name: "Bob".to_string(),
            age: 25,
        };

        let h1 = arena.fetch_if_or_append(alice1.clone(), 1, |p1, p2| p1.name == p2.name);
        let h2 = arena.fetch_if_or_append(alice2.clone(), 1, |p1, p2| p1.name == p2.name);
        assert_eq!(h1, h2); // Same name, should reuse h1
        assert_eq!(arena[h1].age, 30);

        let h3 = arena.fetch_if_or_append(bob.clone(), 1, |p1, p2| p1.name == p2.name);
        assert_ne!(h1, h3);
        assert_eq!(arena[h3].name, "Bob");
    }

    #[test]
    fn test_clear() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 5);
        let h2 = arena.append(2, 5);
        assert_eq!(arena.len(), 10);
        arena.clear();
        assert!(arena.is_empty());
        assert!(arena.spans().is_empty());
        assert!(arena.free_spans().is_empty());
        assert!(arena.try_get(h1).is_err());
        assert!(arena.try_get(h2).is_err());
    }

    #[test]
    fn free_and_reuse_partially() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 10); // 0..10
        arena.free(h1); // Free 0..10
        let h2 = arena.append(2, 3); // Reuse 0..3
        let h3 = arena.append(3, 4); // Reuse 3..7
        assert_eq!(h2.index(), 0);
        assert_eq!(h3.index(), 3);
        assert_eq!(arena.spans(), &[Span::new(0, 3), Span::new(3, 7)]);
        assert_eq!(arena.free_spans(), &[Span::new(7, 10)]);
        assert_eq!(arena[h2], 2);
        assert_eq!(arena[h3], 3);
    }

    #[test]
    fn handle_ordering() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 1);
        let h2 = arena.append(2, 1);
        assert!(h1 < h2);
        arena.free(h1);
        let h3 = arena.append(3, 1); // Should reuse h1's spot
        assert_eq!(h3, h1);
        assert!(h3 < h2);
    }

    #[test]
    fn large_counts() {
        let mut arena: Arena<u8> = Arena::new();
        let h1 = arena.append(1, 1000);
        assert_eq!(arena.len(), 1000);
        for i in 0..1000 {
            let handle = unsafe { Handle::from_usize_unchecked(h1.index() + i) };
            assert_eq!(arena[handle], 1);
        }
        arena.free(h1);
        assert!(arena.is_empty());
    }
}
