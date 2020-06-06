extern crate proptest;
extern crate rand;
use crate::low_level::BitShift;
use crate::BigInt;
use proptest::prelude::*;
use proptest::{strategy, test_runner};

#[derive(Debug)]
pub struct ExponentiallyWeighted();
impl Strategy for ExponentiallyWeighted{
    type Tree = proptest::num::u64::BinarySearch;
    type Value = u64;

    fn new_tree(&self, runner: &mut test_runner::TestRunner) -> strategy::NewTree<Self> {
        // Every power of 256 is half as likely as the last
        let start : f64 = runner.rng().sample(rand::distributions::Exp::new(2f64.ln()/256.0));
        Ok(proptest::num::u64::BinarySearch::new(start as u64))
    }
}

pub fn biased_digit() -> impl Strategy<Value = u64> {
    proptest::strategy::Union::new(vec![
        ExponentiallyWeighted{}.boxed(),
        ExponentiallyWeighted{}.prop_map(|x| 0xffffffffffffffff - x).boxed(),
        any::<u64>().boxed(),
    ])
}

pub fn biased_digits(range: std::ops::Range<usize>) -> impl Strategy<Value = Vec<u64>> {
    proptest::collection::vec(biased_digit(), range)
}

pub fn any_bigint(range: std::ops::Range<usize>) -> impl Strategy<Value = BigInt> {
    (
        biased_digits(range),
        any::<bool>(),
    )
        .prop_map(|(digits, negative)| BigInt { digits, negative }.normalize())
}
pub fn nonnegative_bigint(range: std::ops::Range<usize>) -> impl Strategy<Value = BigInt> {
    biased_digits(range).prop_map(|digits| {
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
