#![feature(test)]
extern crate packed_simd;
use packed_simd::u128x4;
#[cfg(test)]
extern crate proptest;
#[cfg(test)]
extern crate test;

#[derive(PartialEq, Eq)]
pub struct BigInt {
    digits: Vec<u64>,
}
impl std::fmt::Debug for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BigInt")
            .field("digit", &format!("{:x?}", &self.digits))
            .finish()
    }
}

impl BigInt {
    fn trim(mut self) -> Self {
        while self.digits.last() == Some(&0) {
            self.digits.pop();
        }
        self
    }
}

fn mul_u64(x: u64, y: u64) -> (u64, u64) {
    let merged = (x as u128) * (y as u128);
    (merged as u64, (merged >> 64) as u64)
}

fn add_to_digits(x: u64, digits: &mut [u64]) {
    let (res, overflow) = digits[0].overflowing_add(x);
    digits[0] = res;
    if overflow {
        add_to_digits(1, &mut digits[1..]);
    }
}

fn add_u128_to_digits(x: u128, digits: &mut [u64]) {
    let lsb = digits[0] as u128 + ((digits[1] as u128) << 64);
    let (res, overflow) = lsb.overflowing_add(x);
    digits[0] = res as u64;
    digits[1] = (res >> 64) as u64;
    if overflow {
        add_to_digits(1, &mut digits[2..])
    }
}

fn add_u128_to_digits_with_carry(x: u128, digit0: &mut u64, digit1: &mut u64) -> bool {
    let lsb = *digit0 as u128 + ((*digit1 as u128) << 64);
    let (res, overflow) = lsb.overflowing_add(x);
    *digit0 = res as u64;
    *digit1 = (res >> 64) as u64;
    overflow
}

pub fn schoolbook_mul(l: &BigInt, r: &BigInt) -> BigInt {
    let mut digits = vec![0; l.digits.len() + r.digits.len() + 1];
    for (i, &l_digit) in l.digits.iter().enumerate() {
        let mut carry: bool = false;
        let mut digits_iter = digits[i..].iter_mut();
        let mut digit0 = digits_iter.next().unwrap();
        for (&r_digit, digit1) in r.digits.iter().zip(digits_iter) {
            let prod = (l_digit as u128) * (r_digit as u128) + ((carry as u128) << 64);
            carry = add_u128_to_digits_with_carry(prod, digit0, digit1);
            digit0 = digit1;
        }
        if carry {
            add_to_digits(1, &mut digits[i + r.digits.len()..]);
        }
    }
    BigInt { digits }.trim()
}

pub fn schoolbook_mul_vec(l: &BigInt, r: &BigInt) -> BigInt {
    let mut digits = vec![0; l.digits.len() + r.digits.len() + 1];
    for (i, l_vec) in all_vectors(&l.digits).enumerate() {
        for (j, r_vec) in non_overlapping_vectors(&r.digits).enumerate() {
            let prod = r_vec * l_vec;
            for k in 0..4 {
                let prod_k = prod.extract(k);
                // Since we zero-pad
                if prod_k != 0 {
                    let l_index = i + k - 3;
                    let r_index = 4 * j + k;
                    let digits_ix = l_index + r_index;
                    add_u128_to_digits(prod.extract(k), &mut digits[digits_ix..]);
                }
            }
        }
    }
    BigInt { digits }.trim()
}

fn non_overlapping_vectors<'a>(slice: &'a [u64]) -> impl Iterator<Item = u128x4> + 'a {
    slice.chunks(4).map(|chunk| {
        let mut arr = [0; 4];
        for (chunk_val, arr_val) in chunk.iter().zip(arr.iter_mut()) {
            *arr_val = *chunk_val as u128;
        }
        u128x4::from_slice_unaligned(&arr)
    })
}
fn all_vectors(slice: &[u64]) -> AllVectors {
    if slice.len() == 0 {
        // Skip it!
        AllVectors { slice, end: 4 }
    } else {
        AllVectors { slice, end: 1 }
    }
}

struct AllVectors<'a> {
    slice: &'a [u64],
    end: usize,
}
impl AllVectors<'_> {
    fn get_zero_padded(&self, i: usize) -> u128 {
        self.slice.get(i).copied().unwrap_or(0) as u128
    }
}
impl<'a> Iterator for AllVectors<'a> {
    type Item = u128x4;
    fn next(&mut self) -> Option<Self::Item> {
        let out = if self.end < 4 {
            let mut arr = [0; 4];
            for i in 0..self.end {
                arr[i + 4 - self.end] = self.get_zero_padded(i) as u128;
            }
            u128x4::from_slice_unaligned(&arr)
        } else if self.end > self.slice.len() + 3 {
            return None;
        } else if self.end > self.slice.len() {
            let mut arr = [0; 4];
            for i in self.end - 4..self.slice.len() {
                arr[i + 4 - self.end] = self.get_zero_padded(i) as u128;
            }
            u128x4::from_slice_unaligned(&arr)
        } else {
            let mut arr = [0; 4];
            for i in self.end - 4..self.end {
                arr[i + 4 - self.end] = self.get_zero_padded(i) as u128;
            }
            u128x4::from_slice_unaligned(&arr)
        };
        self.end += 1;
        Some(out)
    }
}
pub fn add(l: &BigInt, r: &BigInt) -> BigInt {
    let (BigInt { digits: big }, BigInt { digits: small }) = if l.digits.len() > r.digits.len() {
        (l, r)
    } else {
        (r, l)
    };
    let mut digits = big.clone();
    // TODO: prevent possible allocation here?
    digits.push(0);
    for (i, &x) in small.iter().enumerate() {
        add_to_digits(x, &mut digits[i..]);
    }
    BigInt { digits }.trim()
}

#[cfg(test)]
mod tests {
    extern crate cpuprofiler;
    extern crate proptest;
    extern crate rand;
    extern crate rand_chacha;
    use crate::*;
    use cpuprofiler::PROFILER;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use std::collections::HashMap;
    use test::Bencher;
    fn any_bigint(range: std::ops::Range<usize>) -> impl Strategy<Value = BigInt> {
        proptest::collection::vec(any::<u64>(), range).prop_map(|digits| BigInt { digits }.trim())
    }
    #[test]
    fn hardcoded() {
        let a = BigInt { digits: vec![2] };
        let b = BigInt {
            digits: vec![0x8000000000000000, 1],
        };
        let prod = schoolbook_mul(&a, &b);
        let c = BigInt { digits: vec![0, 3] };
        assert_eq!(prod, c);
    }
    proptest! {
        #[test]
        fn mul_small(a in any::<u64>(), b in any::<u64>()) {
            let a_big = BigInt{digits: vec![a]}.trim();
            let b_big = BigInt{digits: vec![b]}.trim();
            let prod = schoolbook_mul(&a_big, &b_big);
            let prod_pair = (prod.digits.get(0).copied().unwrap_or(0), prod.digits.get(1).copied().unwrap_or(0));
            assert_eq!(prod_pair, mul_u64(a, b));
       }
    }
    proptest! {
        #[test]
        fn mul_zero(a in any_bigint(0..20)) {
            let zero = BigInt{digits: vec![]};
            let prod = schoolbook_mul(&zero, &a);
            assert_eq!(prod, zero);
        }
    }
    proptest! {
        #[test]
        fn mul_identity(a in any_bigint(0..20)) {
            let one = BigInt{digits: vec![1]};
            let prod = schoolbook_mul(&one, &a);
            assert_eq!(prod, a);
        }
    }
    proptest! {
        #[test]
        fn distributive(a in any_bigint(0..20),b in any_bigint(0..20),c in any_bigint(0..20)) {
            let sum_first = schoolbook_mul(&add(&a, &b), &c);
            let sum_last = add(&schoolbook_mul(&a, &c), &schoolbook_mul(&b, &c));
            assert_eq!(sum_first, sum_last);
        }
    }
    #[test]
    fn test_schoolbook_mul_vec_hardcoded() {
        let operands = vec![
            (BigInt { digits: vec![] }, BigInt { digits: vec![] }),
            (BigInt { digits: vec![] }, BigInt { digits: vec![1] }),
            (BigInt { digits: vec![1] }, BigInt { digits: vec![1] }),
            (BigInt { digits: vec![1] }, BigInt { digits: vec![0, 1] }),
            (
                BigInt {
                    digits: vec![0x8c6cd24f9aa81b31, 0xbdbd7388a1e4c9d9],
                },
                BigInt {
                    digits: vec![0xa47022a51237d68c, 0xf482e52c7bc4ac4d],
                },
            ),
            (
                BigInt {
                    digits: vec![0x73fde98ef330eb13],
                },
                BigInt {
                    digits: vec![
                        0x4ca73f50,
                        0x6d6d7b43b33eb7bb,
                        0xef00e9667b95fcd4,
                        0xc03809ed69a7940d,
                        0x8d4b408187ab2453,
                    ],
                },
            ),
        ];
        for (a, b) in operands {
            let expected = schoolbook_mul(&a, &b);
            let actual = schoolbook_mul_vec(&a, &b);
            assert_eq!(expected, actual);
        }
    }
    proptest! {
        #[test]
        fn test_schoolbook_mul_vec(a in any_bigint(0..20),b in any_bigint(0..20)) {
            let expected = schoolbook_mul(&a, &b);
            let actual = schoolbook_mul_vec(&a, &b);
            assert_eq!(expected, actual);
        }
    }
    fn random_bigint(rng: &mut rand_chacha::ChaCha8Rng, size: usize) -> BigInt {
        let mut digits = vec![0; size];
        for x in digits.iter_mut() {
            *x = rng.gen();
        }
        BigInt { digits }
    }
    #[test]
    fn test_all_vectors_hardcoded() {
        let arr = [0, 1, 2, 3, 4];
        let mut iter = all_vectors(&arr);
        let mut tmp = [0; 4];
        iter.next().unwrap().write_to_slice_aligned(&mut tmp);
        assert_eq!(tmp, [0, 0, 0, 0]);
        iter.next().unwrap().write_to_slice_aligned(&mut tmp);
        assert_eq!(tmp, [0, 0, 0, 1]);
        iter.next().unwrap().write_to_slice_aligned(&mut tmp);
        assert_eq!(tmp, [0, 0, 1, 2]);
        iter.next().unwrap().write_to_slice_aligned(&mut tmp);
        assert_eq!(tmp, [0, 1, 2, 3]);
        iter.next().unwrap().write_to_slice_aligned(&mut tmp);
        assert_eq!(tmp, [1, 2, 3, 4]);
        iter.next().unwrap().write_to_slice_aligned(&mut tmp);
        assert_eq!(tmp, [2, 3, 4, 0]);
        iter.next().unwrap().write_to_slice_aligned(&mut tmp);
        assert_eq!(tmp, [3, 4, 0, 0]);
        iter.next().unwrap().write_to_slice_aligned(&mut tmp);
        assert_eq!(tmp, [4, 0, 0, 0]);
        assert_eq!(iter.next(), None);
    }
    #[test]
    fn test_all_vectors_empty() {
        let arr = [];
        let mut iter = all_vectors(&arr);
        assert_eq!(iter.next(), None);
    }
    proptest! {
        #[test]
        fn test_all_vectors(l in 1u64..1000) {
            let test_vector : Vec<u64> = (1..l+1).into_iter().collect();
            let mut counts = HashMap::new();
            for v in all_vectors(&test_vector) {
                for i in 0..4 {
                    *counts.entry(v.extract(i)).or_insert(0) += 1;
                }
            }
            assert_eq!(*counts.get(&0).unwrap(), 12);
            for i in test_vector {
                assert_eq!(*counts.get(&(i as u128)).unwrap(), 4);
            }
        }
    }
    #[bench]
    fn bench_schoolbook_mul(bench: &mut Bencher) {
        PROFILER
            .lock()
            .unwrap()
            .start(format!("profiling/schoolbook_mul.profile"))
            .unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let a = random_bigint(&mut rng, 100);
        let b = random_bigint(&mut rng, 100);
        bench.iter(|| schoolbook_mul(&a, &b));
        PROFILER.lock().unwrap().stop().unwrap();
    }
    #[bench]
    fn bench_schoolbook_mul_vec(bench: &mut Bencher) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let a = random_bigint(&mut rng, 100);
        let b = random_bigint(&mut rng, 100);
        bench.iter(|| schoolbook_mul_vec(&a, &b));
    }
}
