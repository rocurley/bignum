#[cfg(test)]
extern crate proptest;

#[derive(PartialEq, Eq, Debug)]
pub struct BigInt {
    digits: Vec<u64>,
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

pub fn schoolbook_mul(l: &BigInt, r: &BigInt) -> BigInt {
    let mut digits = vec![0; l.digits.len() + r.digits.len() + 1];
    for (i, &l_digit) in l.digits.iter().enumerate() {
        for (j, &r_digit) in r.digits.iter().enumerate() {
            let (lsb, msb) = mul_u64(l_digit, r_digit);
            add_to_digits(lsb, &mut digits[i + j..]);
            add_to_digits(msb, &mut digits[i + j + 1..]);
        }
    }
    BigInt { digits }.trim()
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
    extern crate proptest;
    use crate::{add, mul_u64, schoolbook_mul, BigInt};
    use proptest::prelude::*;
    fn any_bigint(range: std::ops::Range<usize>) -> impl Strategy<Value = BigInt> {
        proptest::collection::vec(any::<u64>(), range).prop_map(|digits| BigInt { digits }.trim())
    }
    proptest! {
        #[test]
        fn mul_small(a in any::<u64>(), b in any::<u64>()) {
            let a_big = BigInt{digits: vec![a]};
            let b_big = BigInt{digits: vec![b]};
            let prod = schoolbook_mul(&a_big, &b_big);
            let prod_pair = (prod.digits[0], prod.digits.get(1).copied().unwrap_or(0));
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
}
