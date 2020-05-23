use crate::low_level::{split_digits_iter, BitShift};
use crate::BigInt;

pub fn fourier_mul(x: &BigInt, y: &BigInt) -> BigInt {
    let output_len = x.digits.len() + y.digits.len() + 1;
    // 100k benchmarks:
    // 3 : [2.0965 s 2.1230 s 2.1747 s]
    // 4 : [1.7231 s 1.7627 s 1.8012 s]
    // 5 : [1.2545 s 1.2652 s 1.2864 s]
    // 6 : [981.68 ms 1.0015 s 1.0270 s]
    // 7 : [721.56 ms 722.83 ms 723.81 ms]
    // 8 : [610.21 ms 617.54 ms 630.83 ms]
    // 9 : [588.68 ms 596.32 ms 612.42 ms]
    //10 : [684.76 ms 696.90 ms 716.33 ms]
    let chunks_exp = 9usize; // TODO: tune this, ideally scale with input size
    let chunks = 2usize.pow(chunks_exp as u32);
    let chunk_size = (output_len + chunks - 1) / chunks;
    // TODO: shouldn't we be dividing chunks_exp by 64?
    let mod_exp = 2 * chunk_size + chunks_exp + 3; //Why 3?
    let p = fourier(mod_exp, chunk_size, chunks_exp, &x);
    let q = fourier(mod_exp, chunk_size, chunks_exp, &y);
    let pq: Vec<BigInt> = p
        .into_iter()
        .zip(q.into_iter())
        .map(|(p, q)| succ_mod_3(&(&p * &q), mod_exp))
        .collect();
    let mut out = inv_fourier(mod_exp, chunk_size, &pq);
    out.negative = x.negative ^ y.negative;
    out.normalize()
}

// Not an FFT, just a reference implementation for testing.
// mod_exp: B = 2^(64*mod_exp) + 1, fourier transform will be mod B.
// chunks_exp: break up into a vector of 2^chunks_exp before FFT.
// chunk_size: how big the chunks will be. Zero-pads as needed. If you want to recover the original
// result, chunk_size should be <= mod_exp.
pub fn fourier(mod_exp: usize, chunk_size: usize, chunks_exp: usize, x: &BigInt) -> Vec<BigInt> {
    if *x == BigInt::ZERO {
        return Vec::new();
    }
    let chunks = 2usize.pow(chunks_exp as u32); // 2^k
    let split: Vec<BigInt> = split_digits_iter(&x.digits, chunk_size)
        .take(chunks)
        .collect();
    let mut out = vec![BigInt::ZERO; chunks];
    fourier_inner_fast::<NormalF>(mod_exp, &split, 1, &mut out);
    out
}

pub fn inv_fourier(mod_exp: usize, chunk_size: usize, p: &[BigInt]) -> BigInt {
    if p.len() == 0 {
        return BigInt::ZERO;
    }
    if p.len().count_ones() != 1 {
        panic!(format!(
            "Expected length of p to be a power of 2, but was instead {}",
            p.len()
        ));
    }
    let chunks_exp = p.len().trailing_zeros();
    let mut out_buffer = vec![BigInt::ZERO; p.len()];
    fourier_inner_fast::<InvF>(mod_exp, p, 1, &mut out_buffer);
    let mut out = BigInt::ZERO;
    for (k, mut x) in out_buffer.into_iter().enumerate() {
        let digits_shift = k * chunk_size;
        x = modular_shift(&x, 2 * 64 * mod_exp - chunks_exp as usize, mod_exp);
        x = succ_mod_2(&x, mod_exp);
        out += &x << BitShift::from_usize(digits_shift * 64);
    }
    out
}

fn fourier_inner_fast<F: FourierMethod>(
    mod_exp: usize,
    xs: &[BigInt],
    stride: usize,
    out: &mut [BigInt],
) {
    // Make the borrow checker happy by doing this before the borrow
    let order = out.len();
    let half_len = out.len() / 2;
    // Benchmarks for 10k*10k fourier_mul:
    // 2: 246,400,190 ns/iter
    // 4: 171,149,792 ns/iter
    // 8:  83,501,405 ns/iter
    // 16: 83,113,803 ns/iter
    // 32: 83,346,466 ns/iter
    if order <= 16 {
        fourier_inner_quadratic::<F>(mod_exp, xs, stride, out);
        return;
    }
    let (left_out, right_out) = out.split_at_mut(half_len);
    fourier_inner_fast::<F>(mod_exp, xs, stride * 2, left_out);
    fourier_inner_fast::<F>(mod_exp, &xs[stride..], stride * 2, right_out);
    for (i, (l, r)) in left_out.iter_mut().zip(right_out.iter_mut()).enumerate() {
        // Want:
        // l = l + r g^i
        // r = l + r g^(i + half_len)
        // TODO: we could avoid an allocation here by re-using temp, clearing it every loop
        let mut new_r = l.clone();
        add_fourier_term(
            &mut new_r,
            &r,
            F::pow(i + half_len, 1, order),
            order,
            mod_exp,
        );
        add_fourier_term(l, &r, F::pow(i, 1, order), order, mod_exp);
        *r = new_r;
    }
}

fn fourier_inner_quadratic<F: FourierMethod>(
    mod_exp: usize,
    xs: &[BigInt],
    stride: usize,
    out: &mut [BigInt],
) {
    let order = out.len();
    for (k, acc) in out.iter_mut().enumerate() {
        for i in 0..order {
            add_fourier_term(acc, &xs[i * stride], F::pow(i, k, order), order, mod_exp);
        }
    }
}

trait FourierMethod {
    fn pow(i: usize, k: usize, order: usize) -> usize;
}

struct NormalF {}
impl FourierMethod for NormalF {
    fn pow(i: usize, k: usize, order: usize) -> usize {
        (i % order) * (k % order)
    }
}
struct InvF {}
impl FourierMethod for InvF {
    fn pow(i: usize, k: usize, order: usize) -> usize {
        ((order - i) % order) * (k % order)
    }
}

fn modular_shift(x: &BigInt, shift: usize, mod_exp: usize) -> BigInt {
    // We can apply the mod extremely cheaply:
    // B = 2^N' + 1
    // x = x0 + x1 (B-1) + x2 (B-1)^2 + x3 (B-1)^3
    // x = x0 - x1 + x2 - x3
    // Only one of these will be populated at all, so it will be +- a power of two.
    let shift_bits = shift % (64 * mod_exp);
    let negative = ((shift / (64 * mod_exp)) % 2) != 0;
    let shifted = x << BitShift::from_usize(shift_bits);
    if negative {
        -shifted
    } else {
        shifted
    }
}

// Preconditions:
// let B = 2^(64*mod_exp) + 1
// let g = 2^(2*64*mod_exp/order)
// acc should be interpreted as digits, and should be 2*mod_exp+1 long.
// 0 <= acc < B
// order divides mod_exp*64
// order is a power of two
//
// This means that g^order = 1 mod B
// This function will apply:
// acc += g^pow * x mod B
// preserving 0 <= acc < B
fn add_fourier_term(acc: &mut BigInt, x: &BigInt, pow: usize, order: usize, mod_exp: usize) {
    let prim_root_exp = 2 * 64 * mod_exp / order; // g = 2^prim_root_exp
    let root_pow_exp = prim_root_exp * (pow % order); // g^pow = 2^root_pow_exp
    *acc += modular_shift(x, root_pow_exp, mod_exp);
    *acc = succ_mod_2(&acc, mod_exp);
}

// 2^(64n) + 1
fn pow_succ(n: usize) -> BigInt {
    let mut digits = vec![0; n + 1];
    digits[0] = 1;
    digits[n] = 1;
    BigInt {
        digits,
        negative: false,
    }
}

// Computes x mod (2^(64 mod_blocks) + 1). Limited to inputs of length 2*mod_blocks or less.
fn succ_mod_3(x: &BigInt, mod_blocks: usize) -> BigInt {
    if mod_blocks == 0 {
        return BigInt::ZERO;
    }
    // B = 2^N + 1
    // x0 + x1 (B-1) + x2 (B-1)^2
    // x0 - x1 + x2
    debug_assert!(
        x.digits.len() <= mod_blocks * 3,
        "Expected {} <= {}",
        x.digits.len(),
        mod_blocks * 3
    );
    let [x0, x1, x2] = split_digits!(&x.digits, mod_blocks, 3);
    let mut diff = x0 - x1 + x2;
    let b = pow_succ(mod_blocks);
    if diff > b {
        diff -= b.clone();
    }
    if x.negative {
        diff = -diff;
    }
    if diff < BigInt::ZERO {
        diff += b;
    }
    diff
}
// Computes x mod (2^(64 mod_blocks) + 1). Limited to inputs of length 2*mod_blocks or less.
fn succ_mod_2(x: &BigInt, mod_blocks: usize) -> BigInt {
    if mod_blocks == 0 {
        return BigInt::ZERO;
    }
    // B = 2^N + 1
    // x0 + x1 (B-1)
    // x0 - x1 + x1 B
    // x0 - x1
    debug_assert!(
        x.digits.len() <= mod_blocks * 2,
        "Expected {} <= {}",
        x.digits.len(),
        mod_blocks * 2
    );
    let [x0, x1] = split_digits!(&x.digits, mod_blocks, 2);
    let mut diff = x0 - x1;
    if x.negative {
        diff = -diff;
    }
    if diff < BigInt::ZERO {
        diff + pow_succ(mod_blocks)
    } else {
        diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::div::div_exact;
    use crate::schoolbook_mul;
    use crate::test_utils::*;
    use proptest::prelude::*;
    #[derive(Debug)]
    struct FourierInputs {
        x: BigInt,
        chunks_exp: usize,
        mod_exp: usize,
        chunk_size: usize,
    }

    fn horrible_mod(mut x: BigInt, y: &BigInt) -> BigInt {
        if *y <= BigInt::ZERO {
            panic!("Mod by non-positive number")
        }
        if x < BigInt::ZERO {
            let mut shifted = -y.clone();
            while x < shifted {
                shifted = &shifted << BitShift::from_usize(1);
            }
            x -= shifted;
        }
        while x >= *y {
            let mut shifted = y << BitShift::from_usize(1);
            let mut prior = y.clone();
            while x >= shifted {
                std::mem::swap(&mut shifted, &mut prior);
                shifted = &shifted << BitShift::from_usize(2);
            }
            x -= prior;
        }
        x
    }
    fn any_fourier_inputs() -> impl Strategy<Value = FourierInputs> {
        (1u32..5, nonnegative_bigint(0..10)).prop_map(|(chunks_exp, x)| {
            let chunks = 2usize.pow(chunks_exp);
            let chunk_size = (x.digits.len() + chunks - 1) / chunks;
            FourierInputs {
                x,
                chunks_exp: chunks_exp as usize,
                mod_exp: chunk_size,
                chunk_size,
            }
        })
    }
    fn check_fourier_inv(x: BigInt, mod_exp: usize, chunk_size: usize, chunks_exp: usize) {
        let p = fourier(mod_exp, chunk_size, chunks_exp, &x);
        let x2 = inv_fourier(mod_exp, chunk_size, &p);
        assert_eq!(x, x2);
    }
    proptest! {
        #[test]
        fn test_fourier_inv(inputs in any_fourier_inputs()) {
            check_fourier_inv(inputs.x, inputs.mod_exp, inputs.chunk_size, inputs.chunks_exp)
        }
    }
    #[test]
    fn test_fourier_inv_hardcoded() {
        /*
        check_fourier_inv(
            BigInt {
                negative: false,
                digits: vec![0, 1],
            },
            64,
            1,
            1,
        );
        check_fourier_inv(BigInt::from_u64(0x100000001), 1, 1, 2);
        check_fourier_inv(BigInt::from_u64(0x4000000000000001), 1, 1, 2);
        */
        check_fourier_inv(
            BigInt {
                negative: false,
                digits: vec![0, 0, 1],
            },
            1,
            1,
            2,
        );
    }
    #[derive(Debug)]
    struct AddFourierTermInputs {
        acc: BigInt,
        x: BigInt,
        pow: usize,
        order: usize,
        mod_exp: usize,
    }
    fn any_add_fourier_term_inputs() -> impl Strategy<Value = AddFourierTermInputs> {
        ((1u32..5), (1usize..100)).prop_flat_map(|(k, mod_exp_mul)| {
            let order = 2usize.pow(k);
            let mod_exp_divisor_pow = std::cmp::max(6 /*64=2^6*/, k) - 6;
            let mod_exp = 2usize.pow(mod_exp_divisor_pow) * mod_exp_mul;
            (
                0..order,
                any_bigint(mod_exp + 1..mod_exp + 2),
                any_bigint(mod_exp + 1..mod_exp + 2),
            )
                .prop_map(move |(pow, acc_raw, x_raw)| AddFourierTermInputs {
                    acc: succ_mod_2(&acc_raw, mod_exp),
                    x: succ_mod_2(&x_raw, mod_exp),
                    pow,
                    order,
                    mod_exp,
                })
        })
    }
    fn check_add_fourier_term(inputs: AddFourierTermInputs) {
        let mut actual = inputs.acc.clone();
        add_fourier_term(
            &mut actual,
            &inputs.x,
            inputs.pow,
            inputs.order,
            inputs.mod_exp,
        );
        let prim_root_exp = 2 * 64 * inputs.mod_exp / inputs.order; // g = 2^prim_root_exp
        let prim_root = &BigInt::from_u64(1) << BitShift::from_usize(prim_root_exp);
        let mut to_add = inputs.x;
        for _ in 0..inputs.pow {
            to_add = succ_mod_2(&(&to_add * &prim_root), inputs.mod_exp);
        }
        let expected = succ_mod_2(&(inputs.acc + to_add), inputs.mod_exp);
        assert_eq!(actual, expected);
    }
    proptest! {
        #[test]
        fn test_add_fourier_term(inputs in any_add_fourier_term_inputs()) {
            check_add_fourier_term(inputs)
        }
    }
    #[test]
    fn test_add_fourier_term_hardcoded() {
        let test_cases = vec![
            AddFourierTermInputs {
                acc: BigInt::ZERO,
                x: BigInt {
                    negative: false,
                    digits: vec![0, 1],
                },
                pow: 1,
                order: 4,
                mod_exp: 1,
            },
            // B = 0x10000000000000001
            // x = 0x10000000000000000
            // g =         0x100000000 (8 zeros)
            // acc += g^pow * x mod B
            // expected:
            // acc += 0xffffffff00000001
            // actual:
            // acc +=      0x100000000
            // Note that x mod B = -1, so the result is -g, which means "expected" is correct.
            AddFourierTermInputs {
                acc: BigInt::ZERO,
                x: BigInt {
                    negative: false,
                    digits: vec![0, 1],
                },
                pow: 1,
                order: 4,
                mod_exp: 1,
            },
            AddFourierTermInputs {
                acc: BigInt::from_u64(0xffffffff),
                x: BigInt::from_u64(0x100000001),
                pow: 3,
                order: 4,
                mod_exp: 1,
            },
        ];
        for test_case in test_cases {
            check_add_fourier_term(test_case);
        }
    }
    proptest! {
        #[test]
        fn test_horrible_mod_small(a in any::<u64>(), b in (1..u64::MAX)) {
            let expected = BigInt::from_u64(a % b);
            let actual = horrible_mod(BigInt::from_u64(a), &BigInt::from_u64(b));
            assert_eq!(expected, actual);
        }
    }
    proptest! {
        #[test]
        fn test_horrible_mod(a in nonnegative_bigint(0..5), b in positive_bigint(0..3)) {
            let m = horrible_mod(a.clone(), &b);
            let r = div_exact(&(&a - &m), &b);
            let a_reconstructed = (&r * &b) + m;
            assert_eq!(a_reconstructed, a);
        }
    }
    proptest! {
        #[test]
        fn test_succ_mod_2(a in any_bigint(0..10)) {
            let mod_blocks = (a.digits.len() + 1)/2;
            let mod_base = pow_succ(mod_blocks);
            assert_eq!(succ_mod_2(&a, mod_blocks), horrible_mod(a, &mod_base));
        }
    }
    proptest! {
        #[test]
        fn test_succ_mod_3(a in any_bigint(0..10)) {
            let mod_blocks = (a.digits.len() + 2)/3;
            let mod_base = pow_succ(mod_blocks);
            assert_eq!(succ_mod_3(&a, mod_blocks), horrible_mod(a, &mod_base));
        }
    }
    #[test]
    fn test_succ_mod_2_hardcoded() {
        let test_cases = vec![(
            BigInt {
                negative: true,
                digits: vec![1, 1],
            },
            1,
        )];
        for (a, mod_blocks) in test_cases {
            let mod_base = pow_succ(mod_blocks);
            assert_eq!(succ_mod_2(&a, mod_blocks), horrible_mod(a, &mod_base));
        }
    }
    proptest! {
        #[test]
        fn test_fourier_mul(a in any_bigint(0..100),b in any_bigint(0..100)) {
            let expected = schoolbook_mul(&a, &b);
            let actual = fourier_mul(&a, &b);
            assert_eq!(expected, actual);
        }
    }
    #[test]
    fn test_fourier_mul_hardcoded() {
        let test_cases = vec![(
            BigInt {
                negative: false,
                digits: vec![0, 1],
            },
            BigInt {
                negative: false,
                digits: vec![0, 0, 0, 0, 0, 0, 0, 1],
            },
        )];
        for (a, b) in test_cases {
            let expected = schoolbook_mul(&a, &b);
            let actual = fourier_mul(&a, &b);
            assert_eq!(expected, actual);
        }
    }
}
