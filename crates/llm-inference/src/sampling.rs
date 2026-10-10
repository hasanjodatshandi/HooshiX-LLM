//! Turning one position's logits into one token id.
//!
//! Nothing here touches Burn. The sampler operates on a plain slice of scores,
//! which keeps it testable without a model, a device or a backend, and keeps the
//! randomness in one place that can be seeded and replayed.
//!
//! # Randomness is injected
//!
//! The sampler does not reach for a global generator. It takes a
//! [`RandomSource`], so a test can supply a scripted one and a run can supply a
//! seeded one. This also keeps sampling independent of the random source the
//! model was *constructed* with: changing a sampling seed must not change the
//! weights, and reseeding construction must not change a generation.

use crate::error::InferenceError;

/// A source of uniform values in `[0, 1)`.
pub trait RandomSource {
    /// Returns the next value, in `[0, 1)`.
    fn uniform(&mut self) -> f32;
}

/// SplitMix64, seeded explicitly.
///
/// Small, dependency-free, and specified rather than borrowed: a run's
/// generation can be replayed from its seed on any machine. It is not a
/// cryptographic generator and is not used as one.
#[derive(Debug, Clone)]
pub struct SeededRandom {
    state: u64,
}

impl SeededRandom {
    /// Creates a source from a seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl RandomSource for SeededRandom {
    fn uniform(&mut self) -> f32 {
        // SplitMix64: advance the state, then avalanche it.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;

        // The top 24 bits give a value that is exactly representable, so the
        // result is uniform on [0, 1) rather than merely close to it.
        ((z >> 40) as f32) / ((1_u32 << 24) as f32)
    }
}

/// How the next token is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    /// Softmax temperature. Zero means greedy.
    pub temperature: f32,
    /// Keep only the `k` highest-scoring tokens, when set.
    pub top_k: Option<usize>,
    /// Keep the smallest set of tokens whose probabilities sum to `p`, when set.
    pub top_p: Option<f32>,
}

impl SamplingConfig {
    /// Always takes the highest-scoring token.
    #[must_use]
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        }
    }

    /// Rejects settings that cannot be evaluated.
    ///
    /// A negative temperature, a zero `k` or a `p` outside `(0, 1]` would each
    /// produce a distribution that is either undefined or degenerate, and the
    /// failure would surface as strange text rather than as an error.
    pub fn validate(&self) -> Result<(), InferenceError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(InferenceError::InvalidSampling {
                reason: format!(
                    "temperature must be a finite value of at least zero, got {}",
                    self.temperature
                ),
            });
        }
        if let Some(k) = self.top_k
            && k == 0
        {
            return Err(InferenceError::InvalidSampling {
                reason: "top-k of zero would discard every candidate".to_owned(),
            });
        }
        if let Some(p) = self.top_p
            && (!p.is_finite() || p <= 0.0 || p > 1.0)
        {
            return Err(InferenceError::InvalidSampling {
                reason: format!("top-p must be in (0, 1], got {p}"),
            });
        }
        Ok(())
    }

    /// Whether this configuration always takes the highest-scoring token.
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}

/// Chooses a token id from one position's `logits`.
///
/// The order is fixed and each step is observable: scale by temperature, take a
/// softmax, keep the top `k`, keep the smallest set reaching `p`, then draw.
/// Greedy skips all of it, because a temperature of zero makes the softmax a
/// step function that no amount of filtering changes.
pub fn sample(
    logits: &[f32],
    config: SamplingConfig,
    random: &mut impl RandomSource,
) -> Result<u32, InferenceError> {
    config.validate()?;

    if logits.is_empty() {
        return Err(InferenceError::EmptyVocabulary);
    }
    if logits.iter().any(|score| !score.is_finite()) {
        return Err(InferenceError::InvalidSampling {
            reason: "the model produced a non-finite score".to_owned(),
        });
    }

    if config.is_greedy() {
        return Ok(argmax(logits));
    }

    let mut probabilities = softmax(logits, config.temperature);
    if let Some(k) = config.top_k {
        keep_top_k(&mut probabilities, k);
    }
    if let Some(p) = config.top_p {
        keep_top_p(&mut probabilities, f64::from(p));
    }
    normalise(&mut probabilities);

    Ok(draw(&probabilities, random))
}

/// Index of the largest score, taking the first on a tie.
fn argmax(values: &[f32]) -> u32 {
    let mut best = 0_usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best as u32
}

/// Softmax with a temperature, in `f64` so a large vocabulary does not lose
/// precision before the exponential.
fn softmax(logits: &[f32], temperature: f32) -> Vec<f64> {
    let inverse = 1.0 / f64::from(temperature);
    let scaled: Vec<f64> = logits.iter().map(|x| f64::from(*x) * inverse).collect();

    let maximum = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = scaled.iter().map(|x| (x - maximum).exp()).collect();
    let total: f64 = exponentials.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        // Every candidate underflowed. Falling back to uniform keeps the draw
        // well defined; returning a tensor of zeros would silently become NaN.
        let uniform = 1.0 / logits.len() as f64;
        return vec![uniform; logits.len()];
    }
    exponentials.into_iter().map(|x| x / total).collect()
}

/// Keeps the `k` largest probabilities, zeroing the rest.
fn keep_top_k(probabilities: &mut [f64], k: usize) {
    if k >= probabilities.len() {
        return;
    }

    let mut order: Vec<usize> = (0..probabilities.len()).collect();
    // Descending by probability. `total_cmp` is a total order, so this cannot
    // panic on a NaN the way `partial_cmp().unwrap()` would.
    order.sort_by(|left, right| probabilities[*right].total_cmp(&probabilities[*left]));

    for index in order.into_iter().skip(k) {
        probabilities[index] = 0.0;
    }
}

/// Keeps the smallest set of largest probabilities whose sum reaches `p`.
///
/// The first token past the threshold is always kept, so a `p` smaller than the
/// largest probability still leaves one candidate rather than none.
fn keep_top_p(probabilities: &mut [f64], p: f64) {
    let mut order: Vec<usize> = (0..probabilities.len()).collect();
    order.sort_by(|left, right| probabilities[*right].total_cmp(&probabilities[*left]));

    let mut cumulative = 0.0;
    let mut kept = 0_usize;
    for index in &order {
        cumulative += probabilities[*index];
        kept += 1;
        if cumulative >= p {
            break;
        }
    }

    for index in order.into_iter().skip(kept) {
        probabilities[index] = 0.0;
    }
}

/// Rescales to sum to one, leaving an all-zero vector uniform.
fn normalise(probabilities: &mut [f64]) {
    let total: f64 = probabilities.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        let uniform = 1.0 / probabilities.len() as f64;
        probabilities.fill(uniform);
        return;
    }
    for value in probabilities.iter_mut() {
        *value /= total;
    }
}

/// Draws an index from a normalised distribution.
fn draw(probabilities: &[f64], random: &mut impl RandomSource) -> u32 {
    let target = f64::from(random.uniform());
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if target < cumulative {
            return index as u32;
        }
    }
    // Floating-point accumulation can leave the total a hair below one; the
    // last candidate is the correct answer rather than an error.
    (probabilities.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::{RandomSource, SamplingConfig, SeededRandom, sample};
    use crate::error::InferenceError;

    /// Returns a scripted sequence of values, then keeps repeating the last.
    struct Scripted {
        values: Vec<f32>,
        position: usize,
    }

    impl Scripted {
        fn new(values: Vec<f32>) -> Self {
            Self {
                values,
                position: 0,
            }
        }
    }

    impl RandomSource for Scripted {
        fn uniform(&mut self) -> f32 {
            let value = self.values[self.position.min(self.values.len() - 1)];
            self.position += 1;
            value
        }
    }

    #[test]
    fn greedy_takes_the_largest_score() {
        let mut random = Scripted::new(vec![0.0]);
        let token =
            sample(&[1.0, 5.0, 3.0, 2.0], SamplingConfig::greedy(), &mut random).expect("valid");
        assert_eq!(token, 1);
    }

    #[test]
    fn greedy_ignores_the_random_source() {
        // A temperature of zero is a step function: the draw cannot change it.
        for scripted in [0.0_f32, 0.5, 0.999] {
            let mut random = Scripted::new(vec![scripted]);
            let token =
                sample(&[1.0, 5.0, 3.0], SamplingConfig::greedy(), &mut random).expect("valid");
            assert_eq!(token, 1, "a draw of {scripted} changed a greedy choice");
        }
    }

    #[test]
    fn greedy_breaks_a_tie_by_the_lowest_index() {
        let mut random = Scripted::new(vec![0.0]);
        let token = sample(&[4.0, 4.0, 1.0], SamplingConfig::greedy(), &mut random).expect("valid");
        assert_eq!(token, 0);
    }

    #[test]
    fn top_k_of_one_is_greedy() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: Some(1),
            top_p: None,
        };
        let mut random = Scripted::new(vec![0.99, 0.0]);
        assert_eq!(
            sample(&[1.0, 9.0, 2.0], config, &mut random).expect("valid"),
            1
        );
        assert_eq!(
            sample(&[1.0, 9.0, 2.0], config, &mut random).expect("valid"),
            1
        );
    }

    #[test]
    fn top_k_of_zero_is_rejected() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: Some(0),
            top_p: None,
        };
        let mut random = Scripted::new(vec![0.5]);
        assert!(matches!(
            sample(&[1.0, 2.0], config, &mut random),
            Err(InferenceError::InvalidSampling { .. })
        ));
    }

    #[test]
    fn a_top_p_of_one_keeps_every_candidate() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: Some(1.0),
        };
        // A draw near the top of the range must still be able to select a token
        // that a smaller p would have discarded.
        let mut random = Scripted::new(vec![0.999_999]);
        let token = sample(&[0.0, 0.0, 0.0, 10.0], config, &mut random).expect("valid");
        assert_eq!(token, 3, "p = 1 must not discard the tail");
    }

    #[test]
    fn a_small_top_p_leaves_at_least_one_candidate() {
        // p smaller than the largest probability: the first token past the
        // threshold is still kept, so the draw is defined rather than empty.
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.000_1),
        };
        for draw in [0.0_f32, 0.5, 0.999] {
            let mut random = Scripted::new(vec![draw]);
            let token = sample(&[0.0, 0.0, 20.0], config, &mut random).expect("valid");
            assert_eq!(token, 2);
        }
    }

    #[test]
    fn a_large_temperature_flattens_the_distribution() {
        // With a very high temperature the largest score no longer dominates, so
        // a mid-range draw can reach a token greedy would never take.
        let config = SamplingConfig {
            temperature: 1000.0,
            top_k: None,
            top_p: None,
        };
        let mut random = Scripted::new(vec![0.9]);
        let token = sample(&[0.0, 1.0, 0.0], config, &mut random).expect("valid");
        assert_ne!(token, 1, "a flattened distribution must be able to stray");
    }

    #[test]
    fn a_low_temperature_sharpens_the_distribution() {
        let config = SamplingConfig {
            temperature: 0.01,
            top_k: None,
            top_p: None,
        };
        // A draw of exactly zero lands in the first bucket whatever its width,
        // so the draws start just above it: the claim is about the shape of the
        // distribution, not about the closed end of the interval.
        for draw in [1e-3_f32, 0.5, 0.99] {
            let mut random = Scripted::new(vec![draw]);
            let token = sample(&[0.0, 1.0, 0.0], config, &mut random).expect("valid");
            assert_eq!(token, 1, "a sharp distribution must stay on the peak");
        }
    }

    #[test]
    fn a_negative_temperature_is_rejected() {
        let config = SamplingConfig {
            temperature: -1.0,
            top_k: None,
            top_p: None,
        };
        let mut random = Scripted::new(vec![0.5]);
        assert!(matches!(
            sample(&[1.0, 2.0], config, &mut random),
            Err(InferenceError::InvalidSampling { .. })
        ));
    }

    #[test]
    fn an_out_of_range_top_p_is_rejected() {
        for p in [0.0_f32, -0.5, 1.5, f32::NAN] {
            let config = SamplingConfig {
                temperature: 1.0,
                top_k: None,
                top_p: Some(p),
            };
            let mut random = Scripted::new(vec![0.5]);
            assert!(
                sample(&[1.0, 2.0], config, &mut random).is_err(),
                "top-p {p} should have been rejected"
            );
        }
    }

    #[test]
    fn an_empty_vocabulary_is_reported() {
        let mut random = Scripted::new(vec![0.5]);
        assert!(matches!(
            sample(&[], SamplingConfig::greedy(), &mut random),
            Err(InferenceError::EmptyVocabulary)
        ));
    }

    #[test]
    fn a_non_finite_score_is_reported() {
        let mut random = Scripted::new(vec![0.5]);
        assert!(matches!(
            sample(&[1.0, f32::NAN], SamplingConfig::greedy(), &mut random),
            Err(InferenceError::InvalidSampling { .. })
        ));
    }

    #[test]
    fn the_seeded_source_is_reproducible_and_stays_in_range() {
        let mut first = SeededRandom::new(7);
        let mut second = SeededRandom::new(7);

        for _ in 0..1000 {
            let value = first.uniform();
            assert!(
                (0.0..1.0).contains(&value),
                "{value} left the half-open unit interval"
            );
            assert_eq!(value, second.uniform(), "the same seed must replay");
        }
    }

    #[test]
    fn different_seeds_produce_different_values() {
        let mut first = SeededRandom::new(1);
        let mut second = SeededRandom::new(2);
        let differences = (0..100)
            .filter(|_| (first.uniform() - second.uniform()).abs() > f32::EPSILON)
            .count();
        assert!(differences > 90, "seeds barely differ: {differences}/100");
    }

    #[test]
    fn every_index_is_reachable_under_a_uniform_score() {
        // A sanity check on the draw: with equal scores over four candidates,
        // all four must be selected across enough samples.
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: None,
        };
        let mut random = SeededRandom::new(0xABCD);
        let mut seen = [false; 4];
        for _ in 0..200 {
            let token = sample(&[0.0, 0.0, 0.0, 0.0], config, &mut random).expect("valid");
            seen[token as usize] = true;
        }
        assert_eq!(seen, [true; 4], "some candidate was never reachable");
    }
}
