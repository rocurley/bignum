extern crate rand;
extern crate rand_chacha;
use bigmul::div::{div_exact, inv_u64};
use bigmul::fourier::fourier_mul;
use bigmul::karatsuba::karatsuba_mul;
use bigmul::schoolbook_mul::{schoolbook_mul, schoolbook_mul_asm};
use bigmul::schoolbook_mul_vec::schoolbook_mul_vec;
use bigmul::toom::toom_3;
use bigmul::BigInt;
use cpuprofiler::PROFILER;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use std::path::Path;

struct Profiler {}

impl criterion::profiler::Profiler for Profiler {
    fn start_profiling(&mut self, benchmark_id: &str, _benchmark_dir: &Path) {
        let mut profiler = PROFILER.lock().unwrap();
        let path = format!("profiling/{}.profile", benchmark_id);
        profiler.start(path).unwrap();
    }
    fn stop_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &Path) {
        let mut profiler = PROFILER.lock().unwrap();
        profiler.stop().unwrap();
    }
}

fn random_bigint(rng: &mut rand_chacha::ChaCha8Rng, size: usize) -> BigInt {
    let mut digits = vec![0; size];
    for x in digits.iter_mut() {
        *x = rng.gen();
    }
    let negative = rng.gen();
    BigInt::from_digits(digits, negative)
}
fn bench_schoolbook_mul(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 1000);
    let b = random_bigint(&mut rng, 1000);
    c.bench_function("schoolbook_mul_1k", |bench| {
        bench.iter(|| schoolbook_mul(&a, &b))
    });
}
fn bench_schoolbook_mul_asm(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 1000);
    let b = random_bigint(&mut rng, 1000);
    c.bench_function("schoolbook_mul_asm_1k", |bench| {
        bench.iter(|| schoolbook_mul_asm(&a, &b))
    });
}
fn bench_schoolbook_mul_vec(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 1000);
    let b = random_bigint(&mut rng, 1000);
    c.bench_function("schoolbook_mul_vec_1k", |bench| {
        bench.iter(|| schoolbook_mul_vec(&a, &b))
    });
}
fn bench_karabtsuba_mul(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 1000);
    let b = random_bigint(&mut rng, 1000);
    c.bench_function("karatsuba_mul_vec_1k", |bench| {
        bench.iter(|| karatsuba_mul(&a, &b));
    });
}
fn bench_karabtsuba_mul_10k(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 10000);
    let b = random_bigint(&mut rng, 10000);
    c.bench_function("karatsuba_mul_vec_10k", |bench| {
        bench.iter(|| karatsuba_mul(&a, &b));
    });
}
fn bench_toom_3_10k(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 10000);
    let b = random_bigint(&mut rng, 10000);
    c.bench_function("toom_3_10k", |bench| {
        bench.iter(|| toom_3(&a, &b));
    });
}
fn bench_fourier_mul_10k(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 10000);
    let b = random_bigint(&mut rng, 10000);
    c.bench_function("fourier_mul_10k", |bench| {
        bench.iter(|| fourier_mul(&a, &b));
    });
}
fn bench_fourier_mul_100k(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 100000);
    let b = random_bigint(&mut rng, 100000);
    c.bench_function("fourier_mul_100k", |bench| {
        bench.iter(|| fourier_mul(&a, &b));
    });
}
fn bench_add_assign(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let mut a = random_bigint(&mut rng, 1000);
    let b = random_bigint(&mut rng, 1000);
    c.bench_function("add_assign", |bench| {
        bench.iter(|| a += &b);
    });
}
fn bench_inv_u64(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let mut test_nums = vec![0; 100];
    let mut out = test_nums.clone();
    for x in test_nums.iter_mut() {
        *x = (rng.gen::<u64>() << 1) + 1;
    }
    c.bench_function("inv_u64", |bench| {
        bench.iter(|| {
            for (&x, y) in test_nums.iter().zip(out.iter_mut()) {
                *y = inv_u64(x);
            }
        });
    });
}
fn bench_div_exact(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 1000);
    let b = random_bigint(&mut rng, 1000);
    let prod = &a * &b;
    c.bench_function("div_exact", |bench| {
        bench.iter(|| div_exact(&prod, &a));
    });
}
fn bench_div_six(c: &mut Criterion) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let a = random_bigint(&mut rng, 1000);
    let six = BigInt::from_u64(6);
    let prod = &a * &six;
    c.bench_function("div_six", |bench| {
        bench.iter(|| div_exact(&prod, &six));
    });
}

fn profiled() -> Criterion {
    Criterion::default()
        .sample_size(10)
        //.with_profiler(Profiler {})
}
criterion_group!(
    name = benches;
    config = profiled();
    targets =
        bench_schoolbook_mul,
        bench_schoolbook_mul_asm,
        bench_schoolbook_mul_vec,
        bench_karabtsuba_mul,
        bench_karabtsuba_mul_10k,
        bench_toom_3_10k,
        bench_fourier_mul_10k,
        bench_fourier_mul_100k,
        bench_add_assign,
        bench_inv_u64,
        bench_div_exact,
        bench_div_six,
);
criterion_main!(benches);
