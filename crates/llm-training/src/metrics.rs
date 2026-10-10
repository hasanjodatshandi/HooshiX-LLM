//! What a training run measured.
//!
//! The specification requires a specific set of numbers to be tracked, because a
//! run that is not observed cannot be diagnosed. This module records them; it
//! does not decide what to do about them.

use serde::{Deserialize, Serialize};

/// What one optimizer step measured.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepMetrics {
    /// Optimizer step index, counted from one.
    pub step: u64,
    /// Learning rate applied at this step.
    pub learning_rate: f64,
    /// Mean training loss over the batches that formed the step.
    pub train_loss: f32,
    /// Mean validation loss, when the step included an evaluation.
    pub validation_loss: Option<f32>,
    /// Global gradient norm before clipping, when it was measured.
    pub gradient_norm: Option<f32>,
    /// Tokens consumed by this step.
    pub tokens: u64,
    /// Wall-clock duration of the step.
    pub step_millis: u64,
    /// Whether the step was abandoned instead of applied.
    pub skipped: bool,
}

/// A run's measured history, in step order.
#[derive(Debug, Default, Clone)]
pub struct MetricsLog {
    steps: Vec<StepMetrics>,
}

impl MetricsLog {
    /// Creates an empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one step.
    pub fn record(&mut self, metrics: StepMetrics) {
        self.steps.push(metrics);
    }

    /// Number of recorded steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether nothing has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Every recorded step.
    #[must_use]
    pub fn steps(&self) -> &[StepMetrics] {
        &self.steps
    }

    /// The most recent step.
    #[must_use]
    pub fn last(&self) -> Option<StepMetrics> {
        self.steps.last().copied()
    }

    /// The most recent step, for amendment.
    ///
    /// A validation pass happens after the step it belongs to, so its result is
    /// attached to the record that step already wrote rather than appended as a
    /// separate entry that never saw an optimizer update.
    pub fn latest_mut(&mut self) -> Option<&mut StepMetrics> {
        self.steps.last_mut()
    }

    /// Total tokens consumed, counting only steps that were applied.
    #[must_use]
    pub fn tokens_seen(&self) -> u64 {
        self.steps
            .iter()
            .filter(|step| !step.skipped)
            .map(|step| step.tokens)
            .sum()
    }

    /// How many steps carried a non-finite loss.
    ///
    /// A run may survive one, but a run that keeps producing them is not
    /// training, and this count is what makes that visible.
    #[must_use]
    pub fn non_finite_losses(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| !step.train_loss.is_finite())
            .count()
    }

    /// How many steps were abandoned.
    #[must_use]
    pub fn skipped_steps(&self) -> usize {
        self.steps.iter().filter(|step| step.skipped).count()
    }

    /// Best training loss seen so far.
    #[must_use]
    pub fn best_train_loss(&self) -> Option<f32> {
        self.steps
            .iter()
            .map(|step| step.train_loss)
            .reduce(f32::min)
    }

    /// Mean training loss over the last `window` recorded steps.
    ///
    /// A window rather than the whole history, because a run's early steps are
    /// not representative of where it is now.
    #[must_use]
    pub fn mean_train_loss_over(&self, window: usize) -> Option<f32> {
        if window == 0 {
            return None;
        }
        let start = self.steps.len().saturating_sub(window);
        let recent = &self.steps[start..];
        if recent.is_empty() {
            return None;
        }
        let total: f32 = recent.iter().map(|step| step.train_loss).sum();
        Some(total / recent.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::{MetricsLog, StepMetrics};

    fn step(index: u64, loss: f32, skipped: bool) -> StepMetrics {
        StepMetrics {
            step: index,
            learning_rate: 1e-3,
            train_loss: loss,
            validation_loss: None,
            gradient_norm: None,
            tokens: 1024,
            step_millis: 10,
            skipped,
        }
    }

    #[test]
    fn an_empty_log_reports_nothing_rather_than_zero() {
        let log = MetricsLog::new();
        assert!(log.is_empty());
        assert_eq!(log.best_train_loss(), None);
        assert_eq!(log.mean_train_loss_over(10), None);
        assert_eq!(log.last(), None);
    }

    #[test]
    fn tokens_count_only_applied_steps() {
        let mut log = MetricsLog::new();
        log.record(step(1, 5.0, false));
        log.record(step(2, f32::NAN, true));
        log.record(step(3, 4.0, false));

        assert_eq!(log.len(), 3);
        assert_eq!(log.tokens_seen(), 2048);
        assert_eq!(log.skipped_steps(), 1);
        assert_eq!(log.non_finite_losses(), 1);
    }

    #[test]
    fn the_best_loss_ignores_non_finite_values() {
        let mut log = MetricsLog::new();
        log.record(step(1, 9.0, false));
        log.record(step(2, f32::NAN, true));
        log.record(step(3, 4.0, false));

        // A NaN is neither smaller nor larger, so `min` keeps the finite value.
        assert_eq!(log.best_train_loss(), Some(4.0));
    }

    #[test]
    fn the_windowed_mean_uses_only_the_last_steps() {
        let mut log = MetricsLog::new();
        for (index, loss) in [100.0_f32, 10.0, 2.0, 2.0].into_iter().enumerate() {
            log.record(step(index as u64 + 1, loss, false));
        }

        assert_eq!(log.mean_train_loss_over(2), Some(2.0));
        assert_eq!(log.mean_train_loss_over(4), Some(28.5));
    }

    #[test]
    fn a_window_larger_than_the_history_uses_the_history() {
        let mut log = MetricsLog::new();
        log.record(step(1, 3.0, false));
        log.record(step(2, 5.0, false));

        assert_eq!(log.mean_train_loss_over(100), Some(4.0));
    }

    #[test]
    fn a_zero_window_reports_nothing() {
        let mut log = MetricsLog::new();
        log.record(step(1, 3.0, false));
        assert_eq!(log.mean_train_loss_over(0), None);
    }

    #[test]
    fn the_last_step_is_the_most_recent_one() {
        let mut log = MetricsLog::new();
        log.record(step(1, 3.0, false));
        log.record(step(2, 2.0, false));
        assert_eq!(log.last().map(|entry| entry.step), Some(2));
    }
}
