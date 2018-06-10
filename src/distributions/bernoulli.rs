// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// https://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//! The Bernoulli distribution.

use Rng;
use distributions::Distribution;

/// The Bernoulli distribution.
///
/// This is a special case of the Binomial distribution where `n = 1`.
///
/// # Example
///
/// ```rust
/// use rand::distributions::{Bernoulli, Distribution};
///
/// let d = Bernoulli::new(0.3);
/// let v = d.sample(&mut rand::thread_rng());
/// println!("{} is from a Bernoulli distribution", v);
/// ```
///
/// # Precision
///
/// This `Bernoulli` distribution uses 64 bits from the RNG (a `u64`),
/// so only probabilities that are multiples of 2<sup>-64</sup> can be
/// represented.
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli {
    /// Probability of success, relative to the maximal integer.
    p_int: u64,
}

// To sample from the Bernoulli distribution we use a method that compares a
// random `u64` value `v < (p * 2^64)`.
//
// If `p == 1.0`, the integer `v` to compare against can not represented as a
// `u64`. We manually set it to `u64::MAX` instead (2^64 - 1 instead of 2^64).
// Note that  value of `p < 1.0` can never result in `u64::MAX`, because an
// `f64` only has 53 bits of precision, and the next largest value of `p` will
// result in `2^64 - 2048`.
//
// Also there is a 100% theoretical concern: if someone consistenly wants to
// generate `true` using the Bernoulli distribution (i.e. by using a probability
// of `1.0`), just using `u64::MAX` is not enough. On average it would return
// false once every 2^64 iterations. Some people apparently care about this
// case.
//
// That is why we special-case `u64::MAX` to always return `true`, without using
// the RNG, and pay the performance price for all uses that *are* reasonable.
// Luckily, if `new()` and `sample` are close, the compiler can optimize out the
// extra check.
const ALWAYS_TRUE: u64 = ::core::u64::MAX;

// This is just `2.0.powi(64)`, but written this way because it is not available
// in `no_std` mode.
const SCALE: f64 = 2.0 * (1u64 << 63) as f64;

impl Bernoulli {
    /// Construct a new `Bernoulli` with the given probability of success `p`.
    ///
    /// For `p >= 1.0`, the resulting distribution will always generate true.
    /// For `p <= 0.0`, the resulting distribution will always generate false.
    ///
    /// # Precision
    ///
    /// This method is accurate for any input `p` in the range `[0, 1]` which is
    /// a multiple of 2<sup>-64</sup>. (Note that not all multiples of
    /// 2<sup>-64</sup> in `[0, 1]` can be represented as a `f64`.)
    #[inline]
    pub fn new(p: f64) -> Bernoulli {
        if p < 0.0 {
            Bernoulli { p_int: 0 }
        } else if p >= 1.0 {
            Bernoulli { p_int: ALWAYS_TRUE }
        } else {
            Bernoulli { p_int: (p * SCALE) as u64 }
        }
    }

    /// Construct a new `Bernoulli` with the probability of success of
    /// `numerator`-in-`denominator`. I.e. `new_ratio(2, 3)` will return
    /// a `Bernoulli` with a 2-in-3 chance, or about 67%, of returning `true`.
    ///
    /// If `numerator == denominator` then the returned `Bernoulli` will always
    /// return `true`. If `numerator == 0` it will always return `false`.
    ///
    /// # Panics
    ///
    /// If `denominator == 0` or `numerator > denominator`.
    ///
    #[inline]
    pub fn from_ratio(numerator: u32, denominator: u32) -> Bernoulli {
        assert!(numerator <= denominator, "Bernoulli::from_ratio called with numerator > denominator");
        assert!(denominator > 0, "Bernoulli::from_ratio called with denominator == 0");
        if numerator == denominator {
            return Bernoulli { p_int: ::core::u64::MAX }
        }
        let p_int = ((numerator as f64 / denominator as f64) * SCALE) as u64;
        Bernoulli { p_int }
    }
}

impl Distribution<bool> for Bernoulli {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        // Make sure to always return true for p = 1.0.
        if self.p_int == ALWAYS_TRUE { return true; }
        let v: u64 = rng.gen();
        v < self.p_int
    }
}

#[cfg(test)]
mod test {
    use Rng;
    use distributions::Distribution;
    #[cfg(feature="std")] use std::panic::catch_unwind;
    use super::Bernoulli;

    #[test]
    fn test_trivial() {
        let mut r = ::test::rng(1);
        let always_false1 = Bernoulli::new(0.0);
        let always_true1 = Bernoulli::new(1.0);
        let always_false2 = Bernoulli::new(-1.2);
        let always_true2 = Bernoulli::new(1.1);
        let always_false3 = Bernoulli::from_ratio(0, 5);
        let always_true3 = Bernoulli::from_ratio(3, 3);
        for _ in 0..5 {
            assert_eq!(r.sample(&always_false1), false);
            assert_eq!(r.sample(&always_true1), true);
            assert_eq!(always_false1.sample(&mut r), false);
            assert_eq!(always_true1.sample(&mut r), true);
            assert_eq!(r.sample(&always_false2), false);
            assert_eq!(r.sample(&always_true2), true);
            assert_eq!(always_false2.sample(&mut r), false);
            assert_eq!(always_true2.sample(&mut r), true);
            assert_eq!(r.sample(&always_false3), false);
            assert_eq!(r.sample(&always_true3), true);
            assert_eq!(always_false3.sample(&mut r), false);
            assert_eq!(always_true3.sample(&mut r), true);
        }
    }

    #[test]
    fn test_average() {
        const P: f64 = 0.3;
        const NUM: u32 = 3;
        const DENOM: u32 = 10;
        let d1 = Bernoulli::new(P);
        let d2 = Bernoulli::from_ratio(NUM, DENOM);
        const N: u32 = 100_000;

        let mut sum1: u32 = 0;
        let mut sum2: u32 = 0;
        let mut rng = ::test::rng(2);
        for _ in 0..N {
            if d1.sample(&mut rng) {
                sum1 += 1;
            }
            if d2.sample(&mut rng) {
                sum2 += 1;
            }
        }
        let avg1 = (sum1 as f64) / (N as f64);
        assert!((avg1 - P).abs() < 5e-3);

        let avg2 = (sum2 as f64) / (N as f64);
        assert!((avg2 - (NUM as f64)/(DENOM as f64)).abs() < 5e-3);
    }


    #[test]
    #[cfg(all(feature="std",
              not(target_arch = "wasm32"),
              not(target_arch = "asmjs")))]
    fn test_assertions() {
        assert!(catch_unwind(|| Bernoulli::from_ratio(5, 2)).is_err());
        assert!(catch_unwind(|| Bernoulli::from_ratio(5, 0)).is_err());
        assert!(catch_unwind(|| Bernoulli::from_ratio(0, 0)).is_err());
    }
}
