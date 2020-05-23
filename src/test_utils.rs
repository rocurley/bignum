extern crate proptest;
use crate::low_level::BitShift;
use crate::BigInt;
use proptest::prelude::*;
pub fn any_bigint(range: std::ops::Range<usize>) -> impl Strategy<Value = BigInt> {
    (
        proptest::collection::vec(any::<u64>(), range),
        any::<bool>(),
    )
        .prop_map(|(digits, negative)| BigInt { digits, negative }.normalize())
}
pub fn nonnegative_bigint(range: std::ops::Range<usize>) -> impl Strategy<Value = BigInt> {
    proptest::collection::vec(any::<u64>(), range).prop_map(|digits| {
        BigInt {
            digits,
            negative: false,
        }
        .normalize()
    })
}
pub fn positive_bigint(range: std::ops::Range<usize>) -> impl Strategy<Value = BigInt> {
    nonnegative_bigint(range).prop_map(|num| {
        if num == BigInt::ZERO {
            BigInt::from_u64(1)
        } else {
            num
        }
    })
}
pub fn any_bitshift(range: std::ops::Range<usize>) -> impl Strategy<Value = BitShift> {
    (range, 0u8..64).prop_map(|(digits, bits)| BitShift { digits, bits })
}
