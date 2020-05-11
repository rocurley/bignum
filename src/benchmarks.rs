use super::*;
use crate::div::{div_exact, inv_u64};
use crate::schoolbook_mul_vec::schoolbook_mul_vec;
use crate::test_utils::*;
use cpuprofiler::PROFILER;
use rand::{Rng, SeedableRng};
use test::Bencher;
#[bench]
fn bench_schoolbook_mul(bench: &mut Bencher) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 1000);
    let b = random_bigint(&mut rng, 1000);
    let mut profiler = PROFILER.lock().unwrap();
    profiler
        .start(format!("profiling/schoolbook_mul.profile"))
        .unwrap();
    bench.iter(|| schoolbook_mul(&a, &b));
    profiler.stop().unwrap();
}
#[bench]
fn bench_schoolbook_mul_vec(bench: &mut Bencher) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 1000);
    let b = random_bigint(&mut rng, 1000);
    let mut profiler = PROFILER.lock().unwrap();
    profiler
        .start(format!("profiling/schoolbook_mul_vec.profile"))
        .unwrap();
    bench.iter(|| schoolbook_mul_vec(&a, &b));
    profiler.stop().unwrap();
}
#[bench]
fn bench_karabtsuba_mul(bench: &mut Bencher) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 1000);
    let b = random_bigint(&mut rng, 1000);
    bench.iter(|| karatsuba_mul(&a, &b));
}
#[bench]
fn bench_karabtsuba_mul_10k(bench: &mut Bencher) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 10000);
    let b = random_bigint(&mut rng, 10000);
    let mut profiler = PROFILER.lock().unwrap();
    profiler
        .start(format!("profiling/karabtsuba.profile"))
        .unwrap();
    bench.iter(|| karatsuba_mul(&a, &b));
    profiler.stop().unwrap();
}
#[bench]
fn bench_toom_3_10k(bench: &mut Bencher) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 10000);
    let b = random_bigint(&mut rng, 10000);
    let mut profiler = PROFILER.lock().unwrap();
    profiler.start(format!("profiling/toom3.profile")).unwrap();
    bench.iter(|| toom_3(&a, &b));
    profiler.stop().unwrap();
}
#[bench]
fn bench_add_assign(bench: &mut Bencher) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let mut a = random_bigint(&mut rng, 1000);
    let b = random_bigint(&mut rng, 1000);
    bench.iter(|| a += &b);
}
#[bench]
fn bench_inv_u64(bench: &mut Bencher) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let mut test_nums = vec![0; 100];
    let mut out = test_nums.clone();
    for x in test_nums.iter_mut() {
        *x = (rng.gen::<u64>() << 1) + 1;
    }
    bench.iter(|| {
        for (&x, y) in test_nums.iter().zip(out.iter_mut()) {
            *y = inv_u64(x);
        }
    });
}
#[bench]
fn bench_div_exact(bench: &mut Bencher) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 1000);
    let b = random_bigint(&mut rng, 1000);
    let prod = &a * &b;
    bench.iter(|| div_exact(&prod, &a));
}
#[bench]
fn bench_div_six(bench: &mut Bencher) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 1000);
    let six = BigInt::from_u64(6);
    let prod = &a * &six;
    bench.iter(|| div_exact(&prod, &six));
}
