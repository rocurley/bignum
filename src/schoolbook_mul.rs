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

    fn mul_u64(x: u64, y: u64) -> (u64, u64) {
        let merged = (x as u128) * (y as u128);
        (merged as u64, (merged >> 64) as u64)
    }

pub fn schoolbook_mul_asm(l: &BigInt, r: &BigInt) -> BigInt {
    const CHUNK_SIZE : usize = 4;
    let mut digits = vec![0; l.digits.len() + r.digits.len() + 1];
    for (i, &l_digit) in l.digits.iter().enumerate() {
        let mut carry: u64 = 0;
        let (mut digit0, remaining_digits) = digits.split_at_mut(i).1.split_first_mut().unwrap();
        for (r_digits, digits_chunk) in r.digits.chunks_exact(CHUNK_SIZE).zip(remaining_digits.chunks_exact_mut(CHUNK_SIZE)) {
            unsafe {
                asm!{"
                    movq 0x00({y0}), %rax
                    mulq {x}
                    movq %rax, {l0}
                    movq %rdx, {h0}
                    movq 0x08({y0}), %rax
                    mulq {x}
                    movq %rax, {l1}
                    movq %rdx, {h1}
                    movq 0x10({y0}), %rax
                    mulq {x}
                    movq %rax, {l2}
                    movq %rdx, {h2}
                    movq 0x18({y0}), %rax
                    mulq {x}
                    movq %rax, {l3}
                    movq %rdx, {h3}

                    add {l0}, {d0}
                    adc {l1}, {d1}
                    adc {l2}, {d2}
                    adc {l3}, {d3}
                    adc $0, {h3}
                    add {carry:r}, {h0}
                    add {h0}, {d1}
                    adc {h1}, {d2}
                    adc {h2}, {d3}
                    adc {h3}, {d4}
                    setb {carry:l}
                ",
                carry = inout(reg) carry,
                x = in(reg) l_digit,
                y0 = in(reg) &r_digits[0],
                l0 = out(reg) _,
                l1 = out(reg) _,
                l2 = out(reg) _,
                l3 = out(reg) _,
                h0 = out(reg) _,
                h1 = out(reg) _,
                h2 = out(reg) _,
                h3 = out(reg) _,
                d0 = inout(reg) *digit0,
                d1 = inout(reg) digits_chunk[0],
                d2 = inout(reg) digits_chunk[1],
                d3 = inout(reg) digits_chunk[2],
                d4 = inout(reg) digits_chunk[3],
                out("rax") _,
                out("rdx") _,
                options(att_syntax),
                };
            }
            digit0 = &mut digits_chunk[3];
        }
        let offset = r.digits.len() - r.digits.len() % 4;
        let mut digits_iter = digits[i+offset..].iter_mut();
        let mut digit0 = digits_iter.next().unwrap();
        for (&r_digit, digit1) in r.digits[offset..].iter().zip(digits_iter) {
            let prod = (l_digit as u128) * (r_digit as u128) + ((carry as u128) << 64);
            carry = add_u128_to_digits_with_carry(prod, digit0, digit1) as u64;
            digit0 = digit1;
        }
        if carry != 0{
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
    proptest! {
        #[test]
        fn asm(a in any_bigint(0..20),b in any_bigint(0..20)) {
            let expected = &a * &b;
            let actual = schoolbook_mul_asm(&a, &b);
            assert_eq!(expected, actual);
        }
    }
}
