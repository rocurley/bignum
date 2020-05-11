use crate::BigInt;

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
pub fn sub_assign_digits(target: &mut [u64], other: &[u64]) {
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
pub fn sub_assign_digits_reverse(target: &mut Vec<u64>, other: &[u64]) {
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
        .map(|digits| BigInt {
            digits: digits.to_vec(),
            negative: false,
        })
        .chain(std::iter::repeat(BigInt::ZERO))
}
