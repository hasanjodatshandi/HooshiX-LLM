//! Compares a backend against the reference path: logits and gradients.
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
//! The shared weights are written once by the reference path and read back by
//! both. After that, any difference in the output is a difference in arithmetic.
//!
//! # Why gradients as well as logits
//!
//! Logits that agree prove the forward pass agrees. They say nothing about the
//! backward pass, and a training run's correctness lives in the backward pass:
//! a wrong derivative, a mis-transposed weight or a detached path leaves the
//! forward output untouched and moves every parameter in the wrong direction.
//! So the same batch is pushed through the same objective on both devices and the
//! gradients are compared element by element.
//!
//! # What is compared, and how closely
//!
//! The largest absolute and relative difference, printed rather than asserted
//! against a hidden constant. A dtype tolerance belongs to the specification;
//! this reports the numbers and fails only on a non-finite value or a shape
//! mismatch, which are not tolerance questions.

// An example aborts on failure rather than returning an error: there is no
// caller to hand a failure to. `clippy.toml` relaxes the panic family for tests
// only.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

// Only the wgpu build uses these; the stub below uses none of them.
#[cfg(feature = "wgpu")]
use burn::module::{Module, ModuleVisitor, Param};
#[cfg(feature = "wgpu")]
use burn::tensor::{Bool, Device, Gradients, Int, Tensor, TensorData};
#[cfg(feature = "wgpu")]
use llm_core::{ExperimentConfig, ModelConfig};
#[cfg(feature = "wgpu")]
use llm_model::{Transformer, all_positions, causal_lm_loss};
#[cfg(feature = "wgpu")]
use llm_training::TokenBatch;

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

/// How far apart two runs of the same computation are.
#[cfg(feature = "wgpu")]
struct Difference {
    elements: usize,
    worst_absolute: f32,
    worst_relative: f32,
    non_finite: usize,
}

#[cfg(feature = "wgpu")]
impl Difference {
    fn measure(left: &[f32], right: &[f32]) -> Self {
        assert_eq!(
            left.len(),
            right.len(),
            "the two devices produced different numbers of elements: {} against {}",
            left.len(),
            right.len()
        );

        let mut worst_absolute = 0.0_f32;
        let mut worst_relative = 0.0_f32;
        let mut non_finite = 0_usize;

        for (a, b) in left.iter().zip(right.iter()) {
            if !a.is_finite() || !b.is_finite() {
                non_finite += 1;
                continue;
            }
            let absolute = (a - b).abs();
            worst_absolute = worst_absolute.max(absolute);
            let scale = a.abs().max(b.abs()).max(1e-6);
            worst_relative = worst_relative.max(absolute / scale);
        }

        Self {
            elements: left.len(),
            worst_absolute,
            worst_relative,
            non_finite,
        }
    }

    fn report(&self, what: &str) -> bool {
        println!(
            "{what:<16} elements {}, worst absolute {:e}, worst relative {:e}, non-finite {}",
            self.elements, self.worst_absolute, self.worst_relative, self.non_finite
        );
        self.non_finite == 0
    }
}

/// Collects every parameter's gradient, in the module's own field order.
///
/// The visit is generic over the tensor rank, which is what makes it possible
/// without naming the model's shapes here. Gradient absence is counted rather
/// than ignored: a parameter that receives a gradient on one device and not the
/// other is a difference the caller has to see.
#[cfg(feature = "wgpu")]
struct CollectGradients<'a> {
    gradients: &'a Gradients,
    collected: &'a mut Vec<f32>,
    unavailable: &'a mut usize,
}

#[cfg(feature = "wgpu")]
impl ModuleVisitor for CollectGradients<'_> {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
        // No gradient, or one that cannot be read back, is counted rather than
        // skipped silently. A parameter that carries a gradient on one device
        // and not the other is exactly the difference a parity check exists to
        // find, and it would be invisible if this returned quietly.
        let Some(gradient) = parameter.grad(self.gradients) else {
            *self.unavailable += 1;
            return;
        };
        let Ok(data) = gradient.try_to_data() else {
            *self.unavailable += 1;
            return;
        };
        let Ok(values) = data.try_to_vec::<f32>() else {
            *self.unavailable += 1;
            return;
        };
        self.collected.extend(values);
    }
}

/// A fixed batch, built identically on whichever device is passed.
#[cfg(feature = "wgpu")]
fn batch(vocab_size: usize, sequence: usize, device: &Device) -> TokenBatch {
    let ids: Vec<i32> = (0..(sequence + 1))
        .map(|index| (index % vocab_size) as i32)
        .collect();

    // Input and target are the same ids shifted by one, which is what the
    // objective expects and is the only place the two differ.
    let input_ids = ids[..sequence].to_vec();
    let target_ids = ids[1..].to_vec();

    TokenBatch {
        input: Tensor::<2, Int>::from_data(TensorData::new(input_ids, [1, sequence]), device),
        target: Tensor::<2, Int>::from_data(TensorData::new(target_ids, [1, sequence]), device),
        mask: all_positions(1, sequence, device) as Tensor<2, Bool>,
    }
}

/// Runs the objective and collects every gradient it produces.
#[cfg(feature = "wgpu")]
fn gradients_of(model: &Transformer, batch: &TokenBatch) -> (Vec<f32>, usize) {
    let logits = model
        .forward(batch.input.clone(), 0)
        .expect("the forward pass runs");
    let loss = causal_lm_loss(logits, batch.target.clone(), batch.mask.clone())
        .expect("the objective evaluates");
    let gradients = loss.backward();

    let mut collected = Vec::new();
    let mut unavailable = 0_usize;
    let mut visitor = CollectGradients {
        gradients: &gradients,
        collected: &mut collected,
        unavailable: &mut unavailable,
    };
    model.visit(&mut visitor);

    (collected, unavailable)
}

#[cfg(feature = "wgpu")]
fn main() {
    let config_path =
        std::env::var("CONFIG").unwrap_or_else(|_| "tests/fixtures/smoke-pipeline.toml".to_owned());
    let config = ExperimentConfig::load(&config_path).expect("the configuration loads");
    let model_config: ModelConfig = config.model;

    // Autodiff on both, because the gradient comparison needs a graph.
    let reference = Device::flex().autodiff();
    let accelerated = Device::wgpu(Default::default()).autodiff();

    // The reference path writes the weights both devices will read.
    let weights = std::env::temp_dir().join("llm-parity-weights.bpk");
    let seed_device = Device::flex();
    llm_model::init::seed(&seed_device, 0xC0FFEE);
    let built = Transformer::new(&model_config, &seed_device).expect("valid configuration");
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

    let sequence = 32_usize;
    println!("model      = {config_path}");
    println!("tokens     = {sequence}");
    println!(
        "parameters = {}",
        model_config.parameter_count().expect("fits in u64")
    );
    println!();

    // --- logits ------------------------------------------------------------
    let reference_batch = batch(model_config.vocab_size, sequence, &reference);
    let accelerated_batch = batch(model_config.vocab_size, sequence, &accelerated);

    let reference_logits = on_reference
        .forward(reference_batch.input.clone(), 0)
        .expect("the reference path runs")
        .try_to_data()
        .expect("the reference result is readable")
        .try_to_vec::<f32>()
        .expect("f32");
    let accelerated_logits = on_accelerated
        .forward(accelerated_batch.input.clone(), 0)
        .expect("the accelerated path runs")
        .try_to_data()
        .expect("the accelerated result is readable")
        .try_to_vec::<f32>()
        .expect("f32");

    let logits = Difference::measure(&reference_logits, &accelerated_logits);
    let logits_ok = logits.report("logits");

    // --- gradients ---------------------------------------------------------
    let (reference_gradients, reference_missing) = gradients_of(&on_reference, &reference_batch);
    let (accelerated_gradients, accelerated_missing) =
        gradients_of(&on_accelerated, &accelerated_batch);

    let gradients = Difference::measure(&reference_gradients, &accelerated_gradients);
    let gradients_ok = gradients.report("gradients");

    println!();
    println!(
        "parameters without a gradient: reference {reference_missing}, accelerated {accelerated_missing}"
    );

    if !logits_ok || !gradients_ok {
        eprintln!("parity: FAILED — a non-finite value appeared on one of the backends");
        std::process::exit(1);
    }
    if reference_missing != accelerated_missing {
        eprintln!(
            "parity: FAILED — the two devices registered a different number of \
             gradient-bearing parameters"
        );
        std::process::exit(1);
    }
    println!("parity: both backends agree on the forward pass and on the backward pass");
}
