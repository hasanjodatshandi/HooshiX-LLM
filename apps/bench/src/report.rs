//! The benchmark report.
//!
//! Every figure is paired with how it was obtained. A report that prints numbers
//! without saying where they came from cannot be re-run, and a number that cannot
//! be re-run is not evidence — which is the whole of section 12 of
//! `11_BENCHMARKING.md`.

use llm_core::{BackendChoice, ExperimentConfig};

use crate::Arguments;

/// What was measured, and on what.
pub struct Report {
    // Environment.
    pub compiled: String,
    pub backend: String,
    pub precision: String,
    pub operating_system: String,
    pub architecture: String,
    pub logical_cpus: usize,
    pub parameters: u64,
    pub model_layers: usize,
    pub model_hidden: usize,
    pub vocab_size: usize,
    // Shapes.
    pub prompt_tokens: usize,
    pub training_steps: usize,
    pub tokens_per_step: u64,
    pub micro_batch: usize,
    pub sequence_length: usize,
    pub accumulation: usize,
    // Measurements.
    pub cold_start_seconds: f64,
    pub ttft_seconds: f64,
    pub prefill_seconds: f64,
    pub decode_seconds: Vec<f64>,
    pub training_seconds: f64,
    pub checkpoint_bytes: u64,
    pub checkpoint_write_seconds: f64,
    pub checkpoint_read_seconds: f64,
}

impl Report {
    /// Captures the environment beside the shapes that will be measured.
    #[must_use]
    pub fn new(config: &ExperimentConfig, arguments: &Arguments, compiled: BackendChoice) -> Self {
        Self {
            backend: config.runtime.backend.to_string(),
            precision: config.runtime.precision.to_string(),
            operating_system: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            logical_cpus: std::thread::available_parallelism().map_or(0, |count| count.get()),
            parameters: config.model.parameter_count().unwrap_or(0),
            model_layers: config.model.num_layers,
            model_hidden: config.model.hidden_size,
            vocab_size: config.model.vocab_size,
            prompt_tokens: arguments.prompt_tokens,
            training_steps: arguments.training_steps,
            tokens_per_step: 0,
            micro_batch: config.data.micro_batch_size,
            sequence_length: config.data.sequence_length,
            accumulation: config.training.gradient_accumulation_steps as usize,
            cold_start_seconds: 0.0,
            ttft_seconds: 0.0,
            prefill_seconds: 0.0,
            decode_seconds: Vec::new(),
            training_seconds: 0.0,
            checkpoint_bytes: 0,
            checkpoint_write_seconds: 0.0,
            checkpoint_read_seconds: 0.0,
            compiled: compiled.to_string(),
        }
    }

    /// Prints the report.
    pub fn print(&self) {
        println!("environment");
        println!("  compiled backend   = {}", self.compiled);
        println!("  configured backend = {}", self.backend);
        println!("  precision          = {}", self.precision);
        println!(
            "  os / arch          = {} / {}",
            self.operating_system, self.architecture
        );
        println!("  logical cpus       = {}", self.logical_cpus);
        println!(
            "  model              = {} layers, hidden {}, vocab {}, {} parameters",
            self.model_layers, self.model_hidden, self.vocab_size, self.parameters
        );
        println!(
            "  batch shapes       = micro {} x sequence {}, accumulation {}",
            self.micro_batch, self.sequence_length, self.accumulation
        );
        println!();

        println!("measurements");
        println!("  cold start         = {:.3} s", self.cold_start_seconds);
        println!(
            "  prefill (ttft)     = {:.3} s for {} tokens = {:.1} tokens/s",
            self.prefill_seconds,
            self.prompt_tokens,
            rate(self.prompt_tokens as f64, self.prefill_seconds)
        );
        let decode_total: f64 = self.decode_seconds.iter().sum();
        if let Some(worst) = self.decode_seconds.iter().copied().reduce(f64::max) {
            println!(
                "  decode             = {:.1} tokens/s over {} tokens ({:.4} s/token mean, {:.4} s/token worst)",
                rate(self.decode_seconds.len() as f64, decode_total),
                self.decode_seconds.len(),
                decode_total / self.decode_seconds.len().max(1) as f64,
                worst
            );
        }
        println!(
            "  training           = {:.1} tokens/s over {} steps",
            rate(
                (self.tokens_per_step * self.training_steps as u64) as f64,
                self.training_seconds
            ),
            self.training_steps
        );
        println!(
            "  checkpoint write   = {:.1} MiB/s ({} bytes in {:.3} s)",
            rate(self.checkpoint_bytes as f64, self.checkpoint_write_seconds) / (1024.0 * 1024.0),
            self.checkpoint_bytes,
            self.checkpoint_write_seconds
        );
        println!(
            "  checkpoint read    = {:.1} MiB/s ({} bytes in {:.3} s)",
            rate(self.checkpoint_bytes as f64, self.checkpoint_read_seconds) / (1024.0 * 1024.0),
            self.checkpoint_bytes,
            self.checkpoint_read_seconds
        );
        println!();
        println!("peak memory is not reported here: see the wrapper, which reads the");
        println!("operating system's own PeakWorkingSet64 for this process.");
    }
}

/// `count / seconds`, or zero when the measurement took no time.
fn rate(count: f64, seconds: f64) -> f64 {
    if seconds > 0.0 { count / seconds } else { 0.0 }
}
