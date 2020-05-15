use crate::low_level::{add_assign_digits_slice, shifted_digits, split_digits_iter, BitShift};
use crate::BigInt;
// Not an FFT, just a reference implementation for testing.
pub fn fourier(mod_exp: usize, chunk_size: usize, chunks_exp: usize, x: BigInt) -> Vec<BigInt> {
    let chunks = 2 ^ chunks_exp; // 2^k
    let split: Vec<BigInt> = split_digits_iter(&x.digits, chunk_size)
        .take(2 ^ chunks)
        .collect();
    // TODO: it's unclear if we can/should guarantee that this is a multiple of 64. Doing so would
    // eliminate the shifts entirely (and the resulting allocations), leaving only a call to
    // add_assign_digits_slice. The allocation could in any case be eliminated by removing the call
    // to collect in shifted_digits, and having add_assign_digits_slice take an iterator for other.
    let prim_root_exp = 2 * mod_exp / chunks; // 2N'/(2^k)
    let base_shift = BitShift::from_usize(prim_root_exp);
    (0..chunks)
        .map(|k| {
            // let g=2^prim_root_exp
            // let N'=64*mod_exp
            // let k = chunks_exp
            // we know that g^(2^k) = 1 mod 2^N' + 1
            // we want to compute g^x * y mod 2^N' + 1
            // g^x, mod nothing, will take 4N'/(2^k)x bits. However, we can mod x by 2^k,
            // getting an upper bound of 4N' bits.
            // Before applying any mod (except for the x mod 2^k), we've got a simple power of two.
            // We can apply the mod extremely cheaply:
            // B = 2^N' + 1
            // x = x0 + x1 (B-1) + x2 (B-1)^2 + x3 (B-1)^3
            // x = x0 - x1 + x2 - x3
            // Only one of these will be populated at all, so it will be +- a power of two.
            let mut digits = vec![0; 2 * mod_exp + 1];
            for (i, chunk) in split.iter().enumerate() {
                let pow = prim_root_exp * ((i * k) % chunks);
                let shift = BitShift::from_usize(pow % 64 * mod_exp);
                let negative = (pow / (64 * mod_exp)) % 2;
                add_assign_digits_slice(
                    &mut digits[shift.digits..],
                    shifted_digits(&chunk.digits, shift.bits),
                );
            }
            BigInt {
                digits,
                negative: false,
            }
        })
        .collect()
}

// Preconditions:
// let B = 2^(64*mod_exp) + 1
// let g = 2^(2*64*mod_exp/order)
// acc should be interpreted as digits, and should be 2*mod_exp+1 long.
// 0 <= acc < B
// order divides mod_exp*64
// order is a power of two
//
// This means that k^order = 1 mod B
// This function will apply:
// acc += g^pow * x mod B
// preserving 0 <= acc < B
fn addFourierTerm(acc: &mut BigInt, x: &BigInt, pow: usize, order: usize, mod_exp: usize) {
    let prim_root_exp = 2 * 64 * mod_exp / order; // g = 2^prim_root_exp
    let root_pow_exp = prim_root_exp * (pow % order); // g^pow = 2^root_pow_exp

    // We can apply the mod extremely cheaply:
    // B = 2^N' + 1
    // x = x0 + x1 (B-1) + x2 (B-1)^2 + x3 (B-1)^3
    // x = x0 - x1 + x2 - x3
    // Only one of these will be populated at all, so it will be +- a power of two.
    let shift_bits = pow % (64 * mod_exp);
    let negative = ((pow / (64 * mod_exp)) % 2) != 0;
    // g^pow = +/- 2^shift_bits
    // where +/- is determined by negative
    let shift = BitShift::from_usize(shift_bits);
    let shifted = x >> shift;
    if negative {
        *acc -= shifted;
    } else {
        *acc += shifted;
    }
    *acc = succ_mod(&acc, mod_exp);
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

// Computes x mod (2^(64 mod_blocks) + 1)
fn succ_mod(x: &BigInt, mod_blocks: usize) -> BigInt {
    // B = 2^N + 1
    // x0 + x1 (B-1)
    // x0 - x1 + x1 B
    // x0 - x1
    assert!(x.digits.len() <= mod_blocks * 2);
    let [x0, x1] = split_digits!(&x.digits, mod_blocks, 2);
    let mut diff = x0 - x1;
    diff.negative ^= x.negative;
    if diff < BigInt::ZERO {
        diff + pow_succ(mod_blocks)
    } else {
        diff
    }
}
