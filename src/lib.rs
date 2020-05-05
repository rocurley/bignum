#![feature(test)]
extern crate packed_simd;
use packed_simd::u128x4;
#[cfg(test)]
extern crate proptest;
#[cfg(test)]
extern crate test;

use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

const KARABTSUBA_THRESHOLD: usize = 60;
const TOOM_3_THRESHOLD: usize = 3000;

#[derive(PartialEq, Eq, Clone)]
pub struct BigInt {
    negative: bool,
    digits: Vec<u64>,
}
impl std::fmt::Debug for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BigInt")
            .field("negative", &self.negative)
            .field("digits", &format!("{:x?}", &self.digits))
            .finish()
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        let sign_cmp = other.negative.cmp(&self.negative);
        if sign_cmp != Ordering::Equal {
            return sign_cmp;
        }
        if self.negative {
            other.cmp_abs(self)
        } else {
            self.cmp_abs(other)
        }
    }
}

impl BigInt {
    const ZERO: BigInt = BigInt {
        digits: Vec::new(),
        negative: false,
    };
    fn cmp_abs(&self, other: &Self) -> Ordering {
        let len_cmp = self.digits.len().cmp(&other.digits.len());
        if len_cmp != Ordering::Equal {
            return len_cmp;
        }
        for (s, o) in self.digits.iter().rev().zip(other.digits.iter().rev()) {
            let digit_cmp = s.cmp(o);
            if digit_cmp != Ordering::Equal {
                return digit_cmp;
            }
        }
        Ordering::Equal
    }
    fn normalize_in_place(&mut self) {
        while self.digits.last() == Some(&0) {
            self.digits.pop();
        }
        if self.digits.len() == 0 {
            self.negative = false;
        }
    }
    fn normalize(mut self) -> Self {
        self.normalize_in_place();
        self
    }
    fn neg_in_place(&mut self) {
        if *self != BigInt::ZERO {
            self.negative = !self.negative;
        }
    }
    fn from_u64(x: u64) -> Self {
        BigInt {
            digits: vec![x],
            negative: false,
        }
        .normalize()
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

fn sub_from_digits(x: u64, digits: &mut [u64]) {
    let (res, overflow) = digits[0].overflowing_sub(x);
    digits[0] = res;
    if overflow {
        sub_from_digits(1, &mut digits[1..]);
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
    let negative = l.negative ^ r.negative;
    BigInt { digits, negative }.normalize()
}

fn split_digits_2(digits: &[u64], split: usize) -> [BigInt; 2] {
    let mut iter = split_digits(digits, split);
    [iter.next().unwrap(), iter.next().unwrap()]
}
fn split_digits_3(digits: &[u64], split: usize) -> [BigInt; 3] {
    let mut iter = split_digits(digits, split);
    [
        iter.next().unwrap(),
        iter.next().unwrap(),
        iter.next().unwrap(),
    ]
}
fn split_digits<'a>(digits: &'a [u64], chunk_size: usize) -> impl Iterator<Item = BigInt> + 'a {
    digits
        .chunks(chunk_size)
        .map(|digits| BigInt {
            digits: digits.to_vec(),
            negative: false,
        })
        .chain(std::iter::repeat(BigInt::ZERO))
}

pub fn karatsuba_mul(l: &BigInt, r: &BigInt) -> BigInt {
    if *l == BigInt::ZERO || *r == BigInt::ZERO {
        return BigInt::ZERO;
    }
    let split_len = (std::cmp::max(l.digits.len(), r.digits.len()) + 1) / 2;
    let mut digits = vec![0; l.digits.len() + r.digits.len() + 1];
    let [l0, l1] = split_digits_2(&l.digits, split_len);
    let [r0, r1] = split_digits_2(&r.digits, split_len);
    let prod0 = &l0 * &r0;
    let prod2 = &l1 * &r1;
    let prod1 = &(l0 + l1) * &(r0 + r1) - &prod2 - &prod0;
    add_assign_digits_slice(&mut digits, &prod0.digits);
    add_assign_digits_slice(&mut digits[split_len..], &prod1.digits);
    add_assign_digits_slice(&mut digits[2 * split_len..], &prod2.digits);
    let negative = l.negative ^ r.negative;
    BigInt { digits, negative }.normalize()
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

pub fn shift_combined(a: u64, b: u64, shift: u32) -> u64 {
    let combined = a as u128 + ((b as u128) << 64);
    (combined >> shift) as u64
}

fn shifted_digits(digits: &[u64], shift: u32) -> Vec<u64> {
    if shift == 0 {
        return digits.to_vec();
    }
    match digits.last() {
        None => Vec::new(),
        Some(&last) => digits
            .windows(2)
            .map(|window| shift_combined(window[0], window[1], shift))
            .chain(std::iter::once(last >> shift))
            .collect(),
    }
}

fn cancell_common_pow_twos(a: &BigInt, b: &BigInt) -> (BigInt, BigInt) {
    if *a == BigInt::ZERO || *b == BigInt::ZERO {
        return (a.clone(), b.clone());
    }
    let mut a_digits = &a.digits[..];
    let mut b_digits = &b.digits[..];
    // Strip low zeros
    while let (Some((0, new_a_digits)), Some((0, new_b_digits))) =
        (a_digits.split_first(), b_digits.split_first())
    {
        a_digits = new_a_digits;
        b_digits = new_b_digits;
    }
    let bitshift = std::cmp::min(a_digits[0].trailing_zeros(), b_digits[0].trailing_zeros());
    let a = BigInt {
        negative: a.negative,
        digits: shifted_digits(a_digits, bitshift),
    }
    .normalize();
    let b = BigInt {
        negative: b.negative,
        digits: shifted_digits(b_digits, bitshift),
    }
    .normalize();
    (a, b)
}

fn inv_u64(x: u64) -> u64 {
    if x % 2 == 0 {
        panic!("Attempted to call inv_u64 on even number");
    }
    // We want to compute x^(phi(2^64) - 1), where phi(2^64) is the size of the multiplicative
    // group. We know that x^(phi(2^64)) is 1, because the group is of order phi(2^64). We also
    // know phi(2^64): it's simply the number of odd u64s, which is 2^63.
    let mut y = 1u64;
    let mut x_exp_pow_2 = x;
    for _ in 0..63 {
        y = y.wrapping_mul(x_exp_pow_2);
        x_exp_pow_2 = x_exp_pow_2.wrapping_mul(x_exp_pow_2);
    }
    y
}

pub fn div_exact(num: &BigInt, denom: &BigInt) -> BigInt {
    if *denom == BigInt::ZERO {
        panic!("div_dexact by 0")
    }
    if *num == BigInt::ZERO {
        return BigInt::ZERO;
    }
    let (mut num, denom) = cancell_common_pow_twos(num, denom);
    let mut digits = Vec::<u64>::with_capacity(num.digits.len() - denom.digits.len() + 1);
    let mut num_digits = &mut num.digits[..];
    let leading_denom_inv = inv_u64(denom.digits[0]);
    while let Some(leading) = num_digits.first() {
        let next_digit = leading_denom_inv.wrapping_mul(*leading);
        digits.push(next_digit);
        let prod = &BigInt::from_u64(next_digit) * &denom;
        sub_assign_digits(num_digits, &prod.digits);
        num_digits = &mut num_digits[1..];
    }
    let negative = num.negative ^ denom.negative;
    BigInt { negative, digits }.normalize()
}

impl Neg for BigInt {
    type Output = Self;

    fn neg(mut self) -> Self {
        self.neg_in_place();
        self
    }
}

impl Add for BigInt {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl<'a> Add<&'a BigInt> for BigInt {
    type Output = Self;

    fn add(mut self, other: &'a Self) -> Self {
        self += other;
        self
    }
}

impl<'a> Add<BigInt> for &'a BigInt {
    type Output = BigInt;

    fn add(self, mut other: BigInt) -> BigInt {
        other += self;
        other
    }
}

impl<'a, 'b> Add<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    fn add(self, other: &'b BigInt) -> BigInt {
        let (big, small) = if self.digits.len() > other.digits.len() {
            (self, other)
        } else {
            (other, self)
        };
        big.clone() + small
    }
}

impl AddAssign for BigInt {
    fn add_assign(&mut self, mut other: Self) {
        if self.negative == other.negative {
            if self.digits.len() < other.digits.len() {
                std::mem::swap(self, &mut other);
            }
            add_assign_digits(&mut self.digits, &other.digits);
        } else {
            match self.cmp_abs(&other) {
                Ordering::Greater => {}
                Ordering::Equal => {
                    *self = BigInt::ZERO;
                    return;
                }
                // We could instead use sub_assign_digits_reverse, but this avoids allocating.
                Ordering::Less => std::mem::swap(self, &mut other),
            }
            sub_assign_digits(&mut self.digits, &other.digits)
        }
        self.normalize_in_place();
    }
}

impl<'a> AddAssign<&'a BigInt> for BigInt {
    fn add_assign(&mut self, other: &'a Self) {
        if self.negative == other.negative {
            add_assign_digits(&mut self.digits, &other.digits);
        } else {
            match self.cmp_abs(&other) {
                Ordering::Greater => sub_assign_digits(&mut self.digits, &other.digits),
                Ordering::Equal => {
                    *self = BigInt::ZERO;
                    return;
                }
                Ordering::Less => {
                    sub_assign_digits_reverse(&mut self.digits, &other.digits);
                    self.negative = !self.negative;
                }
            }
        }
        self.normalize_in_place();
    }
}
impl Sub for BigInt {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        self -= other;
        self
    }
}

impl<'a> Sub<&'a BigInt> for BigInt {
    type Output = Self;

    fn sub(mut self, other: &'a Self) -> Self {
        self -= other;
        self
    }
}

impl<'a> Sub<BigInt> for &'a BigInt {
    type Output = BigInt;

    fn sub(self, mut other: BigInt) -> BigInt {
        other -= self;
        -other
    }
}

impl<'a, 'b> Sub<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    fn sub(self, other: &'b BigInt) -> BigInt {
        if self.digits.len() > other.digits.len() {
            let mut out = self.clone();
            out -= other;
            out
        } else {
            let mut out = other.clone();
            out -= self;
            -out
        }
    }
}
impl SubAssign for BigInt {
    fn sub_assign(&mut self, other: Self) {
        *self += -other;
    }
}
impl<'a> SubAssign<&'a BigInt> for BigInt {
    fn sub_assign(&mut self, other: &'a Self) {
        self.neg_in_place();
        *self += other;
        self.neg_in_place();
    }
}

fn add_assign_digits(target: &mut Vec<u64>, other: &[u64]) {
    let target_len = std::cmp::max(target.len(), other.len()) + 1;
    target.resize(target_len, 0);
    add_assign_digits_slice(&mut *target, other);
}

fn add_assign_digits_slice(target: &mut [u64], other: &[u64]) {
    let mut carry = false;
    for (target_digit, &other_digit) in target.iter_mut().zip(other.iter()) {
        let (res, carry1) = target_digit.overflowing_add(carry as u64);
        let (res, carry2) = res.overflowing_add(other_digit);
        *target_digit = res;
        carry = carry1 || carry2;
    }
    if carry {
        add_to_digits(1, &mut target[other.len()..]);
    }
}

// Precondition: target >= other
fn sub_assign_digits(target: &mut [u64], other: &[u64]) {
    let mut borrow = false;
    for (target_digit, &other_digit) in target.iter_mut().zip(other.iter()) {
        let (res, borrow1) = target_digit.overflowing_sub(borrow as u64);
        let (res, borrow2) = res.overflowing_sub(other_digit);
        *target_digit = res;
        borrow = borrow1 || borrow2;
    }
    if borrow {
        sub_from_digits(1, &mut target[other.len()..]);
    }
}
// Precondition: target <= other
fn sub_assign_digits_reverse(target: &mut Vec<u64>, other: &[u64]) {
    target.resize(other.len(), 0);
    let mut borrow = false;
    for (target_digit, &other_digit) in target.iter_mut().zip(other.iter()) {
        let (res, borrow1) = other_digit.overflowing_sub(borrow as u64);
        let (res, borrow2) = res.overflowing_sub(*target_digit);
        *target_digit = res;
        borrow = borrow1 || borrow2;
    }
    assert!(!borrow);
}

impl<'a, 'b> Mul<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    fn mul(self, other: &'b BigInt) -> BigInt {
        let min_len = std::cmp::min(self.digits.len(), other.digits.len());
        if min_len > TOOM_3_THRESHOLD {
            toom_3(self, other)
        } else if min_len > KARABTSUBA_THRESHOLD {
            karatsuba_mul(self, other)
        } else {
            schoolbook_mul(self, other)
        }
    }
}

// toom_3_shuffle applies the inverse of the matrix:
//  1  0  0  0  0
//  1  1  1  1  1
//  1 -1  1 -1  1
//  1  2  4  8 16
//  0  0  0  0  1
fn toom_3_shuffle(rs: [BigInt; 5]) -> [BigInt; 5] {
    let [r1, mut r2, mut r3, mut r4, r5] = rs;
    // dbg!(&r1, &r2, &r3, &r4, &r5);
    r2 -= &r1;
    r3 -= &r1;
    r4 -= &r1;
    std::mem::swap(&mut r2, &mut r4);
    r2 = div_exact(&r2, &BigInt::from_u64(2));
    r3 += &r2;
    r4 -= &r2;
    r3 = div_exact(&r3, &BigInt::from_u64(3));
    r4 += &r3;
    r4 = -div_exact(&r4, &BigInt::from_u64(2));
    r3 -= &r4;
    r3 -= &r5;

    r2 -= &BigInt::from_u64(2) * &r3;
    r2 -= &BigInt::from_u64(4) * &r4;
    r4 -= &BigInt::from_u64(2) * &r5;

    [r1, r2, r3, r4, r5]
}

pub fn toom_3(x: &BigInt, y: &BigInt) -> BigInt {
    if *x == BigInt::ZERO || *y == BigInt::ZERO {
        return BigInt::ZERO;
    }
    let split_len = (std::cmp::max(x.digits.len(), y.digits.len()) + 2) / 3;
    let mut digits = vec![0; x.digits.len() + y.digits.len() + 1];
    let [x0, x1, x2] = split_digits_3(&x.digits, split_len);
    let [y0, y1, y2] = split_digits_3(&y.digits, split_len);
    let r1 = &x0 * &y0; // 0
    let r2 = &(&x0 + &x1 + &x2) * &(&y0 + &y1 + &y2); // 1
    let r3 = &(&x0 - &x1 + &x2) * &(&y0 - &y1 + &y2); // -1
    let r4 = &(&x0 + &BigInt::from_u64(2) * &x1 + &BigInt::from_u64(4) * &x2)
        * &(&y0 + &BigInt::from_u64(2) * &y1 + &BigInt::from_u64(4) * &y2); // 2
    let r5 = &x2 * &y2; // inf
    let ps = toom_3_shuffle([r1, r2, r3, r4, r5]);
    for (i, p) in ps.iter().enumerate() {
        // Slice may be invalid if we multiply a small number by a big one
        if *p != BigInt::ZERO {
            add_assign_digits_slice(&mut digits[i * split_len..], &p.digits);
        }
    }
    let negative = x.negative ^ y.negative;
    BigInt { digits, negative }.normalize()
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
        (
            proptest::collection::vec(any::<u64>(), range),
            any::<bool>(),
        )
            .prop_map(|(digits, negative)| BigInt { digits, negative }.normalize())
    }
    proptest! {
        #[test]
        fn test_addition_methods_match(a in any_bigint(0..20),b in any_bigint(0..20)) {
            let reference_sum = &a + &b;
            assert_eq!(reference_sum, &b + &a);
            assert_eq!(reference_sum, a.clone() + &b);
            assert_eq!(reference_sum, b.clone() + &a);
            assert_eq!(reference_sum, &a + b.clone());
            assert_eq!(reference_sum, &b + a.clone());
            assert_eq!(reference_sum, a.clone() + b.clone());
            assert_eq!(reference_sum, b.clone() + a.clone());
        }
    }
    proptest! {
        #[test]
        fn test_additive_identity(a in any_bigint(0..20)) {
            assert_eq!(a, BigInt::ZERO + &a);
        }
    }
    proptest! {
        #[test]
        fn test_additive_associatvity(
            a in any_bigint(0..20),
            b in any_bigint(0..20),
            c in any_bigint(0..20),
            ) {
            assert_eq!(&a + (&b + &c), (&a + &b) + &c);
        }
    }
    proptest! {
        #[test]
        fn test_add_small(a in any::<u64>(), b in any::<u64>()) {
            let a_big = BigInt{digits: vec![a], negative: false}.normalize();
            let b_big = BigInt{digits: vec![b], negative: false}.normalize();
            let sum_big = a_big + b_big;
            let (sum_small, carry_small) = a.overflowing_add(b);
            assert_eq!(sum_big.digits[0], sum_small);
            if carry_small {
                assert_eq!(sum_big.digits.len(), 2);
                assert_eq!(sum_big.digits[1], 1);
            } else {
                assert_eq!(sum_big.digits.len(), 1);
            }
       }
    }
    proptest! {
        #[test]
        fn test_subtraction_methods_match(a in any_bigint(0..20),b in any_bigint(0..20)) {
            let reference_diff = &a - &b;
            assert_eq!(reference_diff, a.clone() - &b);
            assert_eq!(reference_diff, &a - b.clone());
            assert_eq!(reference_diff, a.clone() - b.clone());
        }
    }
    #[test]
    fn hardcoded() {
        let a = BigInt {
            digits: vec![2],
            negative: false,
        };
        let b = BigInt {
            digits: vec![0x8000000000000000, 1],
            negative: false,
        };
        let prod = schoolbook_mul(&a, &b);
        let c = BigInt {
            digits: vec![0, 3],
            negative: false,
        };
        assert_eq!(prod, c);
    }
    proptest! {
        #[test]
        fn mul_small(a in any::<u64>(), b in any::<u64>()) {
            let a_big = BigInt{digits: vec![a], negative: false}.normalize();
            let b_big = BigInt{digits: vec![b], negative: false}.normalize();
            let prod = schoolbook_mul(&a_big, &b_big);
            let prod_pair = (prod.digits.get(0).copied().unwrap_or(0), prod.digits.get(1).copied().unwrap_or(0));
            assert_eq!(prod_pair, mul_u64(a, b));
       }
    }
    proptest! {
        #[test]
        fn mul_zero(a in any_bigint(0..20)) {
            let prod = schoolbook_mul(&BigInt::ZERO, &a);
            assert_eq!(prod, BigInt::ZERO);
        }
    }
    proptest! {
        #[test]
        fn mul_identity(a in any_bigint(0..20)) {
            let one = BigInt{digits: vec![1], negative: false};
            let prod = schoolbook_mul(&one, &a);
            assert_eq!(prod, a);
        }
    }
    proptest! {
        #[test]
        fn distributive(a in any_bigint(0..20),b in any_bigint(0..20),c in any_bigint(0..20)) {
            let sum_last = schoolbook_mul(&a, &c)+ schoolbook_mul(&b, &c);
            let sum_first = schoolbook_mul(&(a.clone() + b), &c);
            assert_eq!(sum_first, sum_last);
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
    proptest! {
        #[test]
        fn test_schoolbook_mul_vec(a in any_bigint(0..20),b in any_bigint(0..20)) {
            let expected = schoolbook_mul(&a, &b);
            let actual = schoolbook_mul_vec(&a, &b);
            assert_eq!(expected, actual);
        }
    }
    proptest! {
        #[test]
        fn test_karabtsuba_mul(a in any_bigint(0..20),b in any_bigint(0..20)) {
            let expected = schoolbook_mul(&a, &b);
            let actual = karatsuba_mul(&a, &b);
            assert_eq!(expected, actual);
        }
    }
    proptest! {
        #[test]
        fn test_toom_3(a in any_bigint(0..20),b in any_bigint(0..20)) {
            let expected = schoolbook_mul(&a, &b);
            let actual = toom_3(&a, &b);
            assert_eq!(expected, actual);
        }
    }
    #[test]
    fn test_karabtsuba_hardcoded() {
        let operands = vec![(
            BigInt {
                digits: vec![0, 1],
                negative: false,
            },
            BigInt {
                digits: vec![1],
                negative: false,
            },
        )];
        for (a, b) in operands {
            let expected = schoolbook_mul(&a, &b);
            let actual = karatsuba_mul(&a, &b);
            assert_eq!(expected, actual);
        }
    }
    #[test]
    fn test_toom_3_hardcoded() {
        let operands = vec![(
            BigInt {
                digits: vec![0, 0, 0, 1],
                negative: false,
            },
            BigInt {
                digits: vec![1],
                negative: false,
            },
        )];
        for (a, b) in operands {
            let expected = schoolbook_mul(&a, &b);
            let actual = toom_3(&a, &b);
            assert_eq!(expected, actual);
        }
    }
    proptest! {
        #[test]
        fn test_div_exact(a in any_bigint(0..20),b in any_bigint(0..20)) {
            prop_assume!(a != BigInt::ZERO);
            prop_assume!(b != BigInt::ZERO);
            let prod = schoolbook_mul(&a, &b);
            let a_from_div = div_exact(&prod, &b);
            let b_from_div = div_exact(&prod, &a);
            assert_eq!(a, a_from_div);
            assert_eq!(b, b_from_div);
        }
    }
    fn random_bigint(rng: &mut rand_chacha::ChaCha8Rng, size: usize) -> BigInt {
        let mut digits = vec![0; size];
        for x in digits.iter_mut() {
            *x = rng.gen();
        }
        let negative = rng.gen();
        BigInt { digits, negative }
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
    fn test_toom_3_shuffle() {
        fn one() -> BigInt {
            BigInt::from_u64(1)
        }
        fn id(n: usize) -> [BigInt; 5] {
            let mut out = [BigInt::ZERO; 5];
            out[n] = one();
            out
        }
        let col0 = [one(), one(), one(), one(), BigInt::ZERO];
        assert_eq!(toom_3_shuffle(col0), id(0));
        let col1 = [
            BigInt::ZERO,
            one(),
            -one(),
            BigInt::from_u64(2),
            BigInt::ZERO,
        ];
        assert_eq!(toom_3_shuffle(col1), id(1));
        let col2 = [
            BigInt::ZERO,
            one(),
            one(),
            BigInt::from_u64(4),
            BigInt::ZERO,
        ];
        assert_eq!(toom_3_shuffle(col2), id(2));
        let col3 = [
            BigInt::ZERO,
            one(),
            -one(),
            BigInt::from_u64(8),
            BigInt::ZERO,
        ];
        assert_eq!(toom_3_shuffle(col3), id(3));
        let col4 = [BigInt::ZERO, one(), one(), BigInt::from_u64(16), one()];
        assert_eq!(toom_3_shuffle(col4), id(4));
    }
    fn any_odd() -> impl Strategy<Value = u64> {
        any::<u64>().prop_map(|x| (x << 1) + 1)
    }
    #[test]
    fn test_all_vectors_empty() {
        let arr = [];
        let mut iter = all_vectors(&arr);
        assert_eq!(iter.next(), None);
    }
    proptest! {
        #[test]
        fn test_inv_u64(x in any_odd()) {
            let y = inv_u64(x);
            assert_eq!(1, x.wrapping_mul(y));
        }
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
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let a = random_bigint(&mut rng, 1000);
        let b = random_bigint(&mut rng, 1000);
        PROFILER
            .lock()
            .unwrap()
            .start(format!("profiling/schoolbook_mul.profile"))
            .unwrap();
        bench.iter(|| schoolbook_mul(&a, &b));
        PROFILER.lock().unwrap().stop().unwrap();
    }
    #[bench]
    fn bench_schoolbook_mul_vec(bench: &mut Bencher) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let a = random_bigint(&mut rng, 1000);
        let b = random_bigint(&mut rng, 1000);
        PROFILER
            .lock()
            .unwrap()
            .start(format!("profiling/schoolbook_mul_vec.profile"))
            .unwrap();
        bench.iter(|| schoolbook_mul_vec(&a, &b));
        PROFILER.lock().unwrap().stop().unwrap();
    }
    #[bench]
    fn bench_karabtsuba_mul(bench: &mut Bencher) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let a = random_bigint(&mut rng, 1000);
        let b = random_bigint(&mut rng, 1000);
        bench.iter(|| karatsuba_mul(&a, &b));
    }
    #[bench]
    fn bench_karabtsuba_mul_10k(bench: &mut Bencher) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let a = random_bigint(&mut rng, 10000);
        let b = random_bigint(&mut rng, 10000);
        PROFILER
            .lock()
            .unwrap()
            .start(format!("profiling/karabtsuba.profile"))
            .unwrap();
        bench.iter(|| karatsuba_mul(&a, &b));
        PROFILER.lock().unwrap().stop().unwrap();
    }
    #[bench]
    fn bench_toom_3_10k(bench: &mut Bencher) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let a = random_bigint(&mut rng, 10000);
        let b = random_bigint(&mut rng, 10000);
        PROFILER
            .lock()
            .unwrap()
            .start(format!("profiling/toom3.profile"))
            .unwrap();
        bench.iter(|| toom_3(&a, &b));
        PROFILER.lock().unwrap().stop().unwrap();
    }
    #[bench]
    fn bench_add_assign(bench: &mut Bencher) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let mut a = random_bigint(&mut rng, 1000);
        let b = random_bigint(&mut rng, 1000);
        bench.iter(|| a += &b);
    }
    #[bench]
    fn bench_inv_u64(bench: &mut Bencher) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let mut test_nums = vec![0; 100];
        let mut out = test_nums.clone();
        for x in test_nums.iter_mut() {
            *x = (rng.gen::<u64>() << 1) + 1;
        }
        bench.iter(|| {
            for (&x, y) in test_nums.iter().zip(out.iter_mut()) {
                *y = inv_u64(x);
            }
        });
    }
    #[bench]
    fn bench_div_exact(bench: &mut Bencher) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let a = random_bigint(&mut rng, 1000);
        let b = random_bigint(&mut rng, 1000);
        let prod = &a * &b;
        bench.iter(|| div_exact(&prod, &a));
    }
    #[bench]
    fn bench_div_six(bench: &mut Bencher) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let a = random_bigint(&mut rng, 1000);
        let six = BigInt::from_u64(6);
        let prod = &a * &six;
        bench.iter(|| div_exact(&prod, &six));
    }
}
