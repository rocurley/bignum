use crate::div::div_exact;
use crate::low_level::add_assign_digits_slice;
use crate::BigInt;
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
    let [x0, x1, x2] = split_digits!(&x.digits, split_len, 3);
    let [y0, y1, y2] = split_digits!(&y.digits, split_len, 3);
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
            add_assign_digits_slice(&mut digits[i * split_len..], p.digits.iter().copied());
        }
    }
    let negative = x.negative ^ y.negative;
    BigInt { digits, negative }.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schoolbook_mul::schoolbook_mul;
    use crate::test_utils::*;
    use crate::BigInt;
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn test_toom_3(a in any_bigint(0..20),b in any_bigint(0..20)) {
            let expected = schoolbook_mul(&a, &b);
            let actual = toom_3(&a, &b);
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
}
