use crate::low_level::add_u128_to_digits;
use crate::BigInt;
use packed_simd::u128x4;
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
    let negative = l.negative ^ r.negative;
    BigInt { digits, negative }.normalize()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schoolbook_mul::schoolbook_mul;
    use crate::test_utils::*;
    use proptest::prelude::*;
    use std::collections::HashMap;
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
    proptest! {
        #[test]
        fn test_schoolbook_mul_vec(a in any_bigint(0..20),b in any_bigint(0..20)) {
            let expected = schoolbook_mul(&a, &b);
            let actual = schoolbook_mul_vec(&a, &b);
            assert_eq!(expected, actual);
        }
    }
    #[test]
    fn test_schoolbook_mul_vec_hardcoded() {
        let operands = vec![
            (
                BigInt {
                    digits: vec![],
                    negative: false,
                },
                BigInt {
                    digits: vec![],
                    negative: false,
                },
            ),
            (
                BigInt {
                    digits: vec![],
                    negative: false,
                },
                BigInt {
                    digits: vec![1],
                    negative: false,
                },
            ),
            (
                BigInt {
                    digits: vec![1],
                    negative: false,
                },
                BigInt {
                    digits: vec![1],
                    negative: false,
                },
            ),
            (
                BigInt {
                    digits: vec![1],
                    negative: false,
                },
                BigInt {
                    digits: vec![0, 1],
                    negative: false,
                },
            ),
            (
                BigInt {
                    digits: vec![0x8c6cd24f9aa81b31, 0xbdbd7388a1e4c9d9],
                    negative: false,
                },
                BigInt {
                    digits: vec![0xa47022a51237d68c, 0xf482e52c7bc4ac4d],
                    negative: false,
                },
            ),
            (
                BigInt {
                    digits: vec![0x73fde98ef330eb13],
                    negative: false,
                },
                BigInt {
                    digits: vec![
                        0x4ca73f50,
                        0x6d6d7b43b33eb7bb,
                        0xef00e9667b95fcd4,
                        0xc03809ed69a7940d,
                        0x8d4b408187ab2453,
                    ],
                    negative: false,
                },
            ),
        ];
        for (a, b) in operands {
            let expected = schoolbook_mul(&a, &b);
            let actual = schoolbook_mul_vec(&a, &b);
            assert_eq!(expected, actual);
        }
    }
}
