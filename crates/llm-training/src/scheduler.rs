//! The learning-rate schedule: linear warmup, then cosine decay.
//!
//! # Why this is computed rather than scheduled
//!
//! A stateful scheduler has to be persisted in every checkpoint, and a resume
//! that forgets it silently changes the learning rate for the rest of the run.
//! This schedule is a closed-form function of the step number alone, so the
//! step counter *is* the scheduler state. Nothing extra can be lost, and a run
//! resumed from a checkpoint continues on exactly the curve it left.
//!
//! # Shape
//!
//! ```text
//! step < warmup          peak * (step + 1) / warmup
//! warmup <= step < end   peak * (1 + cos(pi * (step - warmup) / cosine_iters)) / 2
//! step >= end            0
//! ```
//!
//! The rate reaches the peak on the last warmup step and returns to zero at the
//! end of the cosine window. A run that continues past the window stays at zero
//! rather than rising again: an unexpected rise late in a run damages a model
//! that was already converged, and the earlier Burn reading of cosine
//! annealing, where the rate climbs back up, is not what this project wants.

use crate::error::TrainingError;

/// A closed-form warmup-then-cosine schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarmupCosineSchedule {
    peak_learning_rate: f64,
    warmup_steps: u64,
    cosine_iters: u64,
}

impl WarmupCosineSchedule {
    /// Creates a schedule.
    ///
    /// Rejects a peak rate that is not positive and finite, and a warmup longer
    /// than the decay window, because neither can be evaluated into a sensible
    /// learning rate.
    pub fn new(
        peak_learning_rate: f64,
        warmup_steps: u64,
        cosine_iters: u64,
    ) -> Result<Self, TrainingError> {
        if !peak_learning_rate.is_finite() || peak_learning_rate <= 0.0 {
            return Err(TrainingError::InvalidSchedule {
                reason: "the peak learning rate must be finite and greater than zero".to_owned(),
            });
        }
        if cosine_iters == 0 {
            return Err(TrainingError::InvalidSchedule {
                reason: "the cosine window must be at least one step".to_owned(),
            });
        }
        if warmup_steps > cosine_iters {
            return Err(TrainingError::InvalidSchedule {
                reason: format!(
                    "warmup ({warmup_steps} steps) must not exceed the cosine window \
                     ({cosine_iters} steps)"
                ),
            });
        }

        Ok(Self {
            peak_learning_rate,
            warmup_steps,
            cosine_iters,
        })
    }

    /// Peak learning rate, reached on the last warmup step.
    #[must_use]
    pub const fn peak_learning_rate(&self) -> f64 {
        self.peak_learning_rate
    }

    /// Number of warmup steps.
    #[must_use]
    pub const fn warmup_steps(&self) -> u64 {
        self.warmup_steps
    }

    /// Length of the cosine window, in steps.
    #[must_use]
    pub const fn cosine_iters(&self) -> u64 {
        self.cosine_iters
    }

    /// First step at which the rate is zero.
    #[must_use]
    pub const fn end_step(&self) -> u64 {
        self.warmup_steps + self.cosine_iters
    }

    /// Learning rate to apply at `step`.
    ///
    /// Well-defined for every step, including one far past the window, so a
    /// caller never has to guard the call.
    #[must_use]
    pub fn learning_rate_at(&self, step: u64) -> f64 {
        if step < self.warmup_steps {
            return self.peak_learning_rate * (step + 1) as f64 / self.warmup_steps as f64;
        }

        let offset = step - self.warmup_steps;
        if offset >= self.cosine_iters {
            return 0.0;
        }

        let fraction = offset as f64 / self.cosine_iters as f64;
        self.peak_learning_rate * (1.0 + (core::f64::consts::PI * fraction).cos()) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::WarmupCosineSchedule;
    use crate::error::TrainingError;

    fn schedule() -> WarmupCosineSchedule {
        WarmupCosineSchedule::new(1e-3, 10, 100).expect("a valid schedule")
    }

    #[test]
    fn a_zero_warmup_starts_at_the_peak() {
        let schedule = WarmupCosineSchedule::new(1e-3, 0, 10).expect("a zero warmup is allowed");
        assert!((schedule.learning_rate_at(0) - 1e-3).abs() < 1e-12);
    }

    #[test]
    fn the_rate_rises_through_warmup_and_reaches_the_peak_at_its_end() {
        let schedule = schedule();
        let mut previous = 0.0;

        for step in 0..schedule.warmup_steps() {
            let rate = schedule.learning_rate_at(step);
            assert!(
                rate > previous,
                "step {step} did not rise: {rate} after {previous}"
            );
            assert!(rate <= schedule.peak_learning_rate());
            previous = rate;
        }

        assert!((previous - schedule.peak_learning_rate()).abs() < 1e-12);
    }

    #[test]
    fn the_rate_decays_monotonically_after_the_peak() {
        let schedule = schedule();
        let mut previous = schedule.learning_rate_at(schedule.warmup_steps());

        for step in schedule.warmup_steps() + 1..schedule.end_step() {
            let rate = schedule.learning_rate_at(step);
            assert!(
                rate < previous,
                "step {step} did not fall: {rate} after {previous}"
            );
            previous = rate;
        }
    }

    #[test]
    fn the_rate_reaches_zero_at_the_end_of_the_window_and_stays_there() {
        let schedule = schedule();

        assert!(schedule.learning_rate_at(schedule.end_step()) < 1e-12);
        assert_eq!(schedule.learning_rate_at(schedule.end_step() + 1), 0.0);
        assert_eq!(schedule.learning_rate_at(u64::MAX), 0.0);
    }

    #[test]
    fn the_halfway_point_is_half_the_peak() {
        let schedule = schedule();
        let halfway = schedule.warmup_steps() + schedule.cosine_iters() / 2;
        let expected = schedule.peak_learning_rate() / 2.0;

        assert!(
            (schedule.learning_rate_at(halfway) - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            schedule.learning_rate_at(halfway)
        );
    }

    #[test]
    fn the_schedule_is_a_pure_function_of_the_step() {
        // The property that makes a checkpoint need no scheduler record.
        let schedule = schedule();
        for step in [0, 5, 10, 60, 109, 110, 500] {
            assert_eq!(
                schedule.learning_rate_at(step),
                schedule.learning_rate_at(step)
            );
        }
    }

    #[test]
    fn a_non_positive_peak_is_rejected() {
        for peak in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                WarmupCosineSchedule::new(peak, 0, 10),
                Err(TrainingError::InvalidSchedule { .. })
            ));
        }
    }

    #[test]
    fn a_zero_window_is_rejected() {
        assert!(matches!(
            WarmupCosineSchedule::new(1e-3, 0, 0),
            Err(TrainingError::InvalidSchedule { .. })
        ));
    }

    #[test]
    fn a_warmup_longer_than_the_window_is_rejected() {
        assert!(matches!(
            WarmupCosineSchedule::new(1e-3, 11, 10),
            Err(TrainingError::InvalidSchedule { .. })
        ));
    }

    #[test]
    fn the_whole_curve_stays_within_the_peak() {
        let schedule = schedule();
        for step in 0..schedule.end_step() + 5 {
            let rate = schedule.learning_rate_at(step);
            assert!(
                (0.0..=schedule.peak_learning_rate()).contains(&rate),
                "step {step} produced {rate}"
            );
        }
    }
}
