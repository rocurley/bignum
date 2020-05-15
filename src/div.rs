use crate::low_level::{shifted_digits, sub_assign_digits};
use crate::BigInt;

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
        sub_assign_digits(num_digits, prod.digits.iter().copied());
        num_digits = &mut num_digits[1..];
    }
    let negative = num.negative ^ denom.negative;
    BigInt { negative, digits }.normalize()
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
    let bitshift = std::cmp::min(a_digits[0].trailing_zeros(), b_digits[0].trailing_zeros()) as u8;
    let a = BigInt {
        negative: a.negative,
        digits: shifted_digits(a_digits, bitshift).collect(),
    }
    .normalize();
    let b = BigInt {
        negative: b.negative,
        digits: shifted_digits(b_digits, bitshift).collect(),
    }
    .normalize();
    (a, b)
}

pub fn inv_u64(x: u64) -> u64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schoolbook_mul;
    use crate::test_utils::*;
    use proptest::prelude::*;
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
    fn any_odd() -> impl Strategy<Value = u64> {
        any::<u64>().prop_map(|x| (x << 1) + 1)
    }
    proptest! {
        #[test]
        fn test_inv_u64(x in any_odd()) {
            let y = inv_u64(x);
            assert_eq!(1, x.wrapping_mul(y));
        }
    }
}
