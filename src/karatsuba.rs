use crate::low_level::add_assign_digits_slice;
use crate::BigInt;
pub fn karatsuba_mul(l: &BigInt, r: &BigInt) -> BigInt {
    if *l == BigInt::ZERO || *r == BigInt::ZERO {
        return BigInt::ZERO;
    }
    let split_len = (std::cmp::max(l.digits.len(), r.digits.len()) + 1) / 2;
    let mut digits = vec![0; l.digits.len() + r.digits.len() + 1];
    let [l0, l1] = split_digits!(&l.digits, split_len, 2);
    let [r0, r1] = split_digits!(&r.digits, split_len, 2);
    let prod0 = &l0 * &r0;
    let prod2 = &l1 * &r1;
    let prod1 = &(l0 + l1) * &(r0 + r1) - &prod2 - &prod0;
    add_assign_digits_slice(&mut digits, &prod0.digits);
    add_assign_digits_slice(&mut digits[split_len..], &prod1.digits);
    add_assign_digits_slice(&mut digits[2 * split_len..], &prod2.digits);
    let negative = l.negative ^ r.negative;
    BigInt { digits, negative }.normalize()
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::schoolbook_mul;
    use crate::test_utils::*;
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn test_karabtsuba_mul(a in any_bigint(0..20),b in any_bigint(0..20)) {
            let expected = schoolbook_mul(&a, &b);
            let actual = karatsuba_mul(&a, &b);
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
}
