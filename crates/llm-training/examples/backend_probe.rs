//! Locates the first operation that stops producing finite values on a backend.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p llm-training --example backend_probe
//! cargo run --release -p llm-training --no-default-features --features cpu --example backend_probe
//! ```
//!
//! The model is built from ordinary operations — a product, a causal mask filled
//! with negative infinity, a softmax over it, rotary rotation, a normalization.
//! A backend that disagrees with the reference on any of them produces a `NaN`
//! loss further up, where the cause is invisible. This walks the same operations
//! in the same order and reports each result, so the disagreement names itself.

// An example aborts on failure rather than returning an error: there is no
// caller to hand a failure to. `clippy.toml` relaxes the panic family for tests
// only.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use burn::tensor::{Device, Int, Tensor, TensorData};
use llm_core::ModelConfig;
use llm_model::{RmsNorm, Rope, Transformer, causal_mask};

#[cfg(feature = "cuda")]
fn compute_device() -> Device {
    Device::cuda(0)
}
#[cfg(feature = "wgpu")]
fn compute_device() -> Device {
    Device::wgpu(0)
}
#[cfg(feature = "cpu")]
fn compute_device() -> Device {
    Device::cpu()
}
#[cfg(feature = "flex")]
fn compute_device() -> Device {
    Device::flex()
}

const BACKEND: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(feature = "wgpu") {
    "wgpu"
} else if cfg!(feature = "cpu") {
    "cpu"
} else {
    "flex"
};

/// Prints a result and whether every value in it is finite.
fn step<const D: usize>(name: &str, tensor: Tensor<D>) {
    let values = tensor
        .to_data()
        .try_to_vec::<f32>()
        .unwrap_or_else(|_| panic!("{name}: could not read the result back"));
    let finite = values.iter().all(|value| value.is_finite());
    let preview: Vec<String> = values
        .iter()
        .take(4)
        .map(|value| format!("{value:.4}"))
        .collect();
    println!(
        "{name:<26} {}  [{}]",
        if finite { "finite    " } else { "NOT FINITE" },
        preview.join(", ")
    );
}

fn main() {
    let device = compute_device();
    println!("backend {BACKEND}");
    println!();

    let ones = Tensor::<2>::ones([4, 4], &device);
    step("ones", ones.clone());
    step("ones.matmul(ones)", ones.clone().matmul(ones));

    // The finiteness check is what training runs before every optimizer step. A
    // backend that gets the comparison or the reduction wrong would make it
    // reject a perfectly good loss, which is exactly the symptom seen here.
    let healthy = Tensor::<1>::from_data(TensorData::new(vec![1.0_f32, 2.0, 3.0], [3]), &device);
    let broken = Tensor::<1>::from_data(TensorData::new(vec![1.0_f32, f32::NAN], [2]), &device);
    println!(
        "{:<26} {} (expected true)",
        "ensure_finite(finite)",
        llm_model::ensure_finite(&healthy, llm_model::NumericalQuantity::Loss).is_ok()
    );
    println!(
        "{:<26} {} (expected false)",
        "ensure_finite(nan)",
        llm_model::ensure_finite(&broken, llm_model::NumericalQuantity::Loss).is_ok()
    );
    println!();

    // The mask is where negative infinity enters the model, and the softmax over
    // it is the first place a backend that mishandles infinities will show it.
    let scores = Tensor::<4>::zeros([1, 2, 3, 3], &device);
    let masked = scores.mask_fill(causal_mask(3, 3, 0, &device), f32::NEG_INFINITY);
    step("causal mask (-inf)", masked.clone());
    step(
        "softmax after mask",
        burn::tensor::activation::softmax(masked.clone(), 3),
    );

    // Rotary rotation calls cos and sin.
    let rope = Rope::new(4, 10_000.0).expect("a valid rope");
    let heads = Tensor::<4>::ones([1, 2, 3, 4], &device);
    match rope.forward(heads.clone(), heads.clone(), 0) {
        Ok((query, _)) => step("rope at position 0", query),
        Err(error) => println!("{:<26} FAILED: {error}", "rope at position 0"),
    }
    match rope.forward(heads.clone(), heads, 7) {
        Ok((query, _)) => step("rope at position 7", query),
        Err(error) => println!("{:<26} FAILED: {error}", "rope at position 7"),
    }

    // A normalization divides by a root mean square.
    let norm = RmsNorm::new(4, 1e-5, &device);
    step("rms_norm", norm.forward(Tensor::<2>::ones([2, 4], &device)));

    // The whole model, which is where the loss went non-finite. `CONFIG` selects
    // the architecture, so the same probe can be run at the tiny size above and
    // at the configured Stage-1 size.
    let config = match std::env::var("CONFIG") {
        Ok(path) => {
            println!("using the architecture from {path}");
            llm_core::ExperimentConfig::load(&path)
                .expect("the configuration loads")
                .model
        }
        Err(_) => ModelConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            ffn_intermediate_size: 32,
            max_sequence_length: 32,
            ..ModelConfig::default()
        },
    };
    llm_model::init::seed(&device, 0xC0FFEE);
    let model = Transformer::new(&config, &device).expect("a valid configuration");

    let tokens =
        Tensor::<2, Int>::from_data(TensorData::new(vec![1_i32, 2, 3, 4], [1, 4]), &device);
    match model.forward(tokens, 0) {
        Ok(logits) => step("full model logits", logits),
        Err(error) => println!("{:<26} FAILED: {error}", "full model logits"),
    }

    // The objective uses a log softmax, which the attention path above does not,
    // and it is the first thing training computes that the forward probe missed.
    println!();
    step(
        "log_softmax",
        burn::tensor::activation::log_softmax(masked, 3),
    );

    // And then the same objective on an autodiff device, so a failure here
    // rather than above points at the backward pass.
    println!();
    println!("on an autodiff device:");
    let training = compute_device().autodiff();
    let training_seed: u64 = std::env::var("SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0xC0FFEE);
    println!("training seed {training_seed}");
    llm_model::init::seed(&training, training_seed);
    let model = Transformer::new(&config, &training).expect("a valid configuration");

    let sequence: usize = std::env::var("SEQ")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    let tokens: Vec<i32> = (0..=sequence).map(|index| (index % 32) as i32).collect();
    println!("training path at sequence {sequence}");
    let tokens = Tensor::<2, Int>::from_data(TensorData::new(tokens, [1, sequence + 1]), &training);
    let (input, target) = llm_model::next_token_pairs(tokens).expect("more than one token");
    let mask = llm_model::all_positions(1, sequence, &training);

    match model.forward(input, 0) {
        Ok(logits) => {
            step("training logits", logits.clone());
            match llm_model::causal_lm_loss(logits, target, mask) {
                Ok(loss) => {
                    step("causal_lm_loss", loss.clone());
                    let gradients = loss.backward();
                    println!("{:<26} computed", "backward");
                    let _ = gradients;
                }
                Err(error) => println!("{:<26} FAILED: {error}", "causal_lm_loss"),
            }
        }
        Err(error) => println!("{:<26} FAILED: {error}", "training logits"),
    }
}
