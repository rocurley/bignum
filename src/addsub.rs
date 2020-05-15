use crate::low_level::{add_assign_digits, sub_assign_digits, sub_assign_digits_reverse};
use crate::BigInt;
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

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
            sub_assign_digits(&mut self.digits, other.digits.iter().copied())
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
                Ordering::Greater => {
                    sub_assign_digits(&mut self.digits, other.digits.iter().copied())
                }
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

impl Neg for BigInt {
    type Output = Self;

    fn neg(mut self) -> Self {
        self.neg_in_place();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use proptest::prelude::*;
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
}
