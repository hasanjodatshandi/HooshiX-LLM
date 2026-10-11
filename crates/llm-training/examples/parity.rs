//! Compares a backend against the reference path.
//!
//! Run with both features enabled, so one process holds two devices:
//!
//! ```text
//! cargo run --release -p llm-training --features flex,wgpu --example parity
//! ```
//!
//! # Why the same weights, not the same seed
//!
//! Two devices seeded identically do **not** produce the same weights: the
//! random source belongs to the device, so the reference path and an accelerator
//! draw different numbers. Comparing two runs that each trained from their own
//! seed would therefore measure the seeds, not the backends.
//!
//! The shared weights are therefore written once by the reference path and read
//! back by both. After that, any difference in the output is a difference in
//! arithmetic.
//!
//! # What is compared, and how closely
//!
//! Logits over the same tokens, element by element, reporting the largest
//! absolute and relative difference. A dtype tolerance belongs to the caller:
//! this prints the numbers and the verdict is a threshold the specification sets,
//! not a constant hidden in a benchmark.

// An example aborts on failure rather than returning an error: there is no
// caller to hand a failure to. `clippy.toml` relaxes the panic family for tests
// only.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

// Only the wgpu build uses these; the stub below uses none of them.
#[cfg(feature = "wgpu")]
use burn::module::Module;
#[cfg(feature = "wgpu")]
use burn::tensor::{Device, Int, Tensor, TensorData};
#[cfg(feature = "wgpu")]
use llm_core::{ExperimentConfig, ModelConfig};
#[cfg(feature = "wgpu")]
use llm_model::Transformer;

/// The comparison, available only when an accelerator backend is compiled in.
///
/// Without it there is nothing to compare against, so the example says so rather
/// than failing to build — `cargo clippy --all-targets` runs the default feature
/// set, and an example that cannot compile there is an example nobody can run
/// the gates with.
#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!(
        "parity: this example compares the reference path against the wgpu backend.\n\
         parity: build it with `--features flex,wgpu`."
    );
}

#[cfg(feature = "wgpu")]
fn main() {
    let config_path =
        std::env::var("CONFIG").unwrap_or_else(|_| "tests/fixtures/smoke-pipeline.toml".to_owned());
    let config = ExperimentConfig::load(&config_path).expect("the configuration loads");
    let model_config: ModelConfig = config.model;

    let reference = Device::flex();
    let accelerated = Device::wgpu(Default::default());

    // The reference path writes the weights both devices will read.
    let weights = std::env::temp_dir().join("llm-parity-weights.bpk");
    llm_model::init::seed(&reference, 0xC0FFEE);
    let built = Transformer::new(&model_config, &reference).expect("valid configuration");
    built
        .clone()
        .save_file(&weights)
        .expect("the reference weights are written");

    let on_reference = Transformer::new(&model_config, &reference).expect("valid configuration");
    let on_reference = on_reference
        .try_load_file(&weights)
        .expect("the reference device reads its own weights");

    let on_accelerated = Transformer::new(&model_config, &accelerated)
        .unwrap_or_else(|error| panic!("the accelerated device cannot build the model: {error}"));
    let on_accelerated = on_accelerated
        .try_load_file(&weights)
        .unwrap_or_else(|error| panic!("the accelerated device cannot read the weights: {error}"));

    // A fixed, in-range input so both devices compute the same thing.
    let sequence = 32_usize;
    let ids: Vec<i32> = (0..sequence)
        .map(|index| (index % model_config.vocab_size) as i32)
        .collect();
    let tokens = |device: &Device| {
        Tensor::<2, Int>::from_data(TensorData::new(ids.clone(), [1, sequence]), device)
    };

    let reference_logits = on_reference
        .forward(tokens(&reference), 0)
        .expect("the reference path runs")
        .to_data()
        .try_to_vec::<f32>()
        .expect("f32");
    let accelerated_logits = on_accelerated
        .forward(tokens(&accelerated), 0)
        .expect("the accelerated path runs")
        .to_data()
        .try_to_vec::<f32>()
        .expect("f32");

    assert_eq!(
        reference_logits.len(),
        accelerated_logits.len(),
        "the two devices produced different shapes"
    );

    let mut worst_absolute = 0.0_f32;
    let mut worst_relative = 0.0_f32;
    let mut non_finite = 0_usize;
    for (left, right) in reference_logits.iter().zip(accelerated_logits.iter()) {
        if !left.is_finite() || !right.is_finite() {
            non_finite += 1;
            continue;
        }
        let absolute = (left - right).abs();
        worst_absolute = worst_absolute.max(absolute);
        let scale = left.abs().max(right.abs()).max(1e-6);
        worst_relative = worst_relative.max(absolute / scale);
    }

    println!("model            = {config_path}");
    println!("tokens           = {sequence}");
    println!("elements         = {}", reference_logits.len());
    println!("worst absolute   = {worst_absolute:e}");
    println!("worst relative   = {worst_relative:e}");
    println!("non-finite pairs = {non_finite}");

    if non_finite > 0 {
        eprintln!("parity: FAILED — one of the backends produced a non-finite value");
        std::process::exit(1);
    }
    println!("parity: both backends produced finite values for the same weights and input");
}
