use crate::BigInt;
use std::cmp::Ordering;
use std::iter::{once, repeat};
use std::ops::{Mul, Shl, Shr};

pub fn add_to_digits(x: u64, digits: &mut [u64]) {
    let (res, overflow) = digits[0].overflowing_add(x);
    digits[0] = res;
    if overflow {
        add_to_digits(1, &mut digits[1..]);
    }
}

pub fn add_u128_to_digits(x: u128, digits: &mut [u64]) {
    let lsb = digits[0] as u128 + ((digits[1] as u128) << 64);
    let (res, overflow) = lsb.overflowing_add(x);
    digits[0] = res as u64;
    digits[1] = (res >> 64) as u64;
    if overflow {
        add_to_digits(1, &mut digits[2..])
    }
}

pub fn sub_from_digits(x: u64, digits: &mut [u64]) {
    let (res, overflow) = digits[0].overflowing_sub(x);
    digits[0] = res;
    if overflow {
        sub_from_digits(1, &mut digits[1..]);
    }
}

pub fn add_u128_to_digits_with_carry(x: u128, digit0: &mut u64, digit1: &mut u64) -> bool {
    let lsb = *digit0 as u128 + ((*digit1 as u128) << 64);
    let (res, overflow) = lsb.overflowing_add(x);
    *digit0 = res as u64;
    *digit1 = (res >> 64) as u64;
    overflow
}

pub fn add_assign_digits(target: &mut Vec<u64>, other: &[u64]) {
    let target_len = std::cmp::max(target.len(), other.len()) + 1;
    target.resize(target_len, 0);
    add_assign_digits_slice(&mut *target, other);
}


pub fn add_assign_digits_slice(target: &mut [u64], other: &[u64]) {
    let mut carry = 0;
    let chunk_size = 6;
    for (target_chunk, other_chunk) in target.chunks_exact_mut(chunk_size).zip(&mut other.chunks_exact(chunk_size)) {
        unsafe {
            asm!{"
                addq {carry:r}, {x0}
                adcq {y0}, {x0}
                adcq {y1}, ({x1})
                adcq {y2}, 8({x1})
                adcq {y3}, 16({x1})
                adcq {y4}, 24({x1})
                adcq {y5}, 32({x1})
                setb {carry:l}
            ",
            carry = inout(reg) carry,
            x0 = inout(reg) target_chunk[0],
            x1 = in(reg) &target_chunk[1],
            y0 = in(reg) other_chunk[0],
            y1 = in(reg) other_chunk[1],
            y2 = in(reg) other_chunk[2],
            y3 = in(reg) other_chunk[3],
            y4 = in(reg) other_chunk[4],
            y5 = in(reg) other_chunk[5],
            options(att_syntax),
            };
        }
    }
    let cleanup_idx = other.len() - other.len() % chunk_size;
    for (target_digit, &other_digit) in target[cleanup_idx..].iter_mut().zip(&other[cleanup_idx..]) {
        unsafe {
            asm!{"
                addq {carry:r}, {x}
                adcq {y}, {x}
                setb {carry:l}
            ",
            carry = inout(reg) carry,
            x = inout(reg) *target_digit,
            y = in(reg) other_digit,
            options(att_syntax),
            };
        }
    }
    if carry > 0 {
        add_to_digits(1, &mut target[other.len()..]);
    }
}

// shift < 64
fn shr_combined(a: u64, b: u64, shift: u8) -> u64 {
    let combined = a as u128 + ((b as u128) << 64);
    (combined >> shift) as u64
}

// shift < 64
fn shl_combined(a: u64, b: u64, shift: u8) -> u64 {
    let combined = a as u128 + ((b as u128) << 64);
    // shift left by shift, but then shift right 64 to take the most significant part
    (combined >> (64 - shift)) as u64
}

// shift < 64
pub fn shr_digits<'a>(digits: &'a [u64], shift: u8) -> Box<dyn Iterator<Item = u64> + 'a> {
    if shift == 0 {
        return Box::new(digits.iter().copied());
    }
    match digits.last() {
        None => Box::new(std::iter::empty()),
        Some(&last) => Box::new(
            digits
                .windows(2)
                .map(move |window| shr_combined(window[0], window[1], shift))
                .chain(std::iter::once(last >> shift)),
        ),
    }
}

// shift < 64
pub fn shl_digits<'a>(digits: &'a [u64], shift: u8) -> Box<dyn Iterator<Item = u64> + 'a> {
    if shift == 0 {
        return Box::new(digits.iter().copied());
    }
    match digits {
        [] => Box::new(std::iter::empty()),
        &[digit] => {
            let combined_shifted = (digit as u128) << shift;
            Box::new(vec![combined_shifted as u64, (combined_shifted >> 64) as u64].into_iter())
        }
        &[first, .., last] => Box::new(
            once(first << shift).chain(
                digits
                    .windows(2)
                    .map(move |window| shl_combined(window[0], window[1], shift))
                    .chain(once(last >> (64 - shift))),
            ),
        ),
    }
}

// Precondition: target >= other
// target = target - other
pub fn sub_assign_digits<I: Iterator<Item = u64>>(target: &mut [u64], other: I) {
    let mut borrow = false;
    let mut len = 0;
    for (target_digit, other_digit) in target.iter_mut().zip(other) {
        len += 1;
        let (res, borrow1) = target_digit.overflowing_sub(borrow as u64);
        let (res, borrow2) = res.overflowing_sub(other_digit);
        *target_digit = res;
        borrow = borrow1 || borrow2;
    }
    if borrow {
        sub_from_digits(1, &mut target[len..]);
    }
}
// Precondition: target <= other
// target = other - target
pub fn sub_assign_digits_reverse(target: &mut Vec<u64>, other: &[u64]) {
    target.resize(other.len(), 0);
    sub_assign_digits_reverse_slice(target, other.iter().copied());
}
// Precondition: target <= other
pub fn sub_assign_digits_reverse_slice<I: Iterator<Item = u64>>(target: &mut [u64], other: I) {
    let mut borrow = false;
    for (target_digit, other_digit) in target.iter_mut().zip(other) {
        let (res, borrow1) = other_digit.overflowing_sub(borrow as u64);
        let (res, borrow2) = res.overflowing_sub(*target_digit);
        *target_digit = res;
        borrow = borrow1 || borrow2;
    }
    assert!(!borrow);
}

#[macro_export]
macro_rules! split_digits {
    ($digits: expr, $split: expr, $n: expr) => {{
        use crate::low_level::split_digits_iter;
        let mut out = [BigInt::ZERO; $n];
        for (chunk, out_chunk) in split_digits_iter($digits, $split).zip(out.iter_mut()) {
            *out_chunk = chunk;
        }
        out
    }};
}

pub fn split_digits_iter<'a>(
    digits: &'a [u64],
    chunk_size: usize,
) -> impl Iterator<Item = BigInt> + 'a {
    digits
        .chunks(chunk_size)
        .map(|digits| {
            BigInt {
                digits: digits.to_vec(),
                negative: false,
            }
            .normalize()
        })
        .chain(std::iter::repeat(BigInt::ZERO))
}

pub fn trim_digits_slice(mut slice: &[u64]) -> &[u64] {
    while let Some((&0, init)) = slice.split_last() {
        slice = init;
    }
    slice
}

pub fn cmp_digits_slice(mut x: &[u64], mut y: &[u64]) -> Ordering {
    x = trim_digits_slice(x);
    y = trim_digits_slice(y);
    let len_cmp = x.len().cmp(&y.len());
    if len_cmp != Ordering::Equal {
        return len_cmp;
    }
    for (s, o) in x.iter().rev().zip(y.iter().rev()) {
        let digit_cmp = s.cmp(o);
        if digit_cmp != Ordering::Equal {
            return digit_cmp;
        }
    }
    Ordering::Equal
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct BitShift {
    pub bits: u8,
    pub digits: usize,
}

impl Mul<usize> for BitShift {
    type Output = BitShift;

    fn mul(self, factor: usize) -> BitShift {
        let bits_unwrapped = self.bits as usize * factor;
        let bits = (bits_unwrapped % 64) as u8;
        let digits =
            self.digits.checked_mul(factor).expect("digits overflowed") + bits_unwrapped / 64;
        BitShift { bits, digits }
    }
}

impl BitShift {
    pub fn from_usize(shift: usize) -> Self {
        let bits = (shift % 64) as u8;
        let digits = shift / 64;
        BitShift { bits, digits }
    }
}

impl Shl<BitShift> for &BigInt {
    type Output = BigInt;
    fn shl(self, shift: BitShift) -> BigInt {
        let digits = repeat(0)
            .take(shift.digits)
            .chain(shl_digits(&self.digits, shift.bits))
            .collect();
        BigInt {
            digits,
            negative: self.negative,
        }
        .normalize()
    }
}

impl Shr<BitShift> for &BigInt {
    type Output = BigInt;
    fn shr(self, shift: BitShift) -> BigInt {
        if shift.digits >= self.digits.len() {
            return BigInt::ZERO;
        }
        let digits = shr_digits(&self.digits[shift.digits..], shift.bits).collect();
        BigInt {
            digits,
            negative: self.negative,
        }
        .normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use proptest::prelude::*;
    fn check_shift_left_right_digits(a: BigInt, shift: u8) {
        let shifted_digits: Vec<u64> = shl_digits(&a.digits, shift).collect();
        let unshifted_digits: Vec<u64> = shr_digits(&shifted_digits, shift).collect();
        let unshifted = BigInt {
            digits: unshifted_digits,
            negative: a.negative,
        }
        .normalize();
        assert_eq!(a, unshifted);
    }
    proptest! {
        #[test]
        fn test_shift_left_right_digits(a in any_bigint(0..20), shift in (0u8..63)) {
            check_shift_left_right_digits(a, shift);
        }
    }
    #[test]
    fn test_shift_left_right_digits_hardcoded() {
        check_shift_left_right_digits(BigInt::from_u64(2), 63);
    }
    fn check_shift_left_right(a: BigInt, shift: BitShift) {
        let unshifted = &(&a << shift) >> shift;
        assert_eq!(a, unshifted);
    }
    proptest! {
        #[test]
        fn test_shift_left_right(a in any_bigint(0..20), shift in any_bitshift(0..20)) {
            check_shift_left_right(a, shift);
        }
    }
    proptest! {
        #[test]
        fn test_shl_digits_mul(a in any_bigint(0..20), shift in (0u8..63)) {
        let digits: Vec<u64> = shl_digits(&a.digits, shift).collect();
        let actual = BigInt{digits, negative: a.negative}.normalize();
        let expected = &a * &BigInt::from_u64(2u64.pow(shift as u32));
        assert_eq!(actual, expected);
        }
    }
    proptest! {
        #[test]
        fn test_shl_mul(a in any_bigint(0..20), shift in (0usize..500)) {
            let actual = &a << BitShift::from_usize(shift);
            let mut expected = a;
            for _ in 0..shift {
                expected = &expected * &BigInt::from_u64(2);
            }
            assert_eq!(actual, expected);
        }
    }
}
