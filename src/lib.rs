#![feature(test)]
extern crate packed_simd;
#[cfg(test)]
extern crate proptest;
#[cfg(test)]
extern crate test;
#[cfg(test)]
mod benchmarks;

use karatsuba::karatsuba_mul;
use schoolbook_mul::schoolbook_mul;
use std::cmp::Ordering;
use std::ops::Mul;
use toom::toom_3;

#[macro_use]
mod low_level;
mod addsub;
pub mod div;
pub mod fourier;
pub mod karatsuba;
pub mod schoolbook_mul;
pub mod schoolbook_mul_vec;
#[cfg(test)]
mod test_utils;
pub mod toom;

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
