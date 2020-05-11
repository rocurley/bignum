use crate::low_level::{add_to_digits, add_u128_to_digits_with_carry};
use crate::BigInt;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use crate::BigInt;
    use proptest::prelude::*;

    fn mul_u64(x: u64, y: u64) -> (u64, u64) {
        let merged = (x as u128) * (y as u128);
        (merged as u64, (merged >> 64) as u64)
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
}
