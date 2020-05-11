extern crate proptest;
use crate::BigInt;
use proptest::prelude::*;
extern crate rand;
extern crate rand_chacha;
use rand::Rng;
pub fn any_bigint(range: std::ops::Range<usize>) -> impl Strategy<Value = BigInt> {
    (
        proptest::collection::vec(any::<u64>(), range),
        any::<bool>(),
    )
        .prop_map(|(digits, negative)| BigInt { digits, negative }.normalize())
}
pub fn random_bigint(rng: &mut rand_chacha::ChaCha8Rng, size: usize) -> BigInt {
    let mut digits = vec![0; size];
    for x in digits.iter_mut() {
        *x = rng.gen();
    }
    let negative = rng.gen();
    BigInt { digits, negative }
}
