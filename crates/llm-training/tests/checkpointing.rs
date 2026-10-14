//! Checkpoint contract tests.
//!
//! A checkpoint that loads but restores the wrong state is worse than no
//! checkpoint, because the run continues and the damage is only visible later.
//! These tests therefore cover three things: that a good checkpoint restores
//! exactly what was stored, that every recognised mismatch is a rejection rather
//! than a warning, and that a resumed run is the run it claims to continue.

// Test helpers may abort on failure: an unusable scratch directory means the
// test cannot run, and the panic message names the cause. `clippy.toml` only
// relaxes the panic family for `#[test]` functions and `#[cfg(test)]` modules,
// which does not reach helpers in an integration test binary. The workspace-level
// denies stay in force for `src/`.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::path::{Path, PathBuf};

use burn::tensor::{Distribution, Int, Tensor, TensorData};
use llm_core::{ModelConfig, OptimizerConfig};
use llm_model::{Transformer, all_positions, next_token_pairs};
use llm_training::checkpoint::{
    self, CHECKPOINT_FORMAT_VERSION, CheckpointExpectation, CheckpointMetadata, DataCursor,
    METADATA_FILE, MODEL_FILE, OPTIMIZER_FILE,
};
use llm_training::{CheckpointError, TokenBatch, Trainer, WarmupCosineSchedule};

const TOKENIZER: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const DATASET: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const CONFIG: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

const VOCABULARY: usize = 256;
const SEQUENCE: usize = 24;
const LEARNING_RATE: f64 = 0.05;

fn tiny_config() -> ModelConfig {
    ModelConfig {
        vocab_size: VOCABULARY,
        hidden_size: 16,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        ffn_intermediate_size: 32,
        max_sequence_length: 32,
        ..ModelConfig::default()
    }
}

fn schedule() -> WarmupCosineSchedule {
    WarmupCosineSchedule::new(LEARNING_RATE, 0, 100).expect("a valid schedule")
}

fn trainer() -> Trainer {
    Trainer::new(schedule(), Some(1.0), &OptimizerConfig::default())
}

/// Serialises everything that draws from the device's random source.
///
/// The reference backend keeps **one process-wide** random source, while cargo
/// runs the tests in a binary as parallel threads. Two tests that seed and draw
/// therefore interleave and observe each other's draws, which makes a seeded
/// comparison meaningless.
///
/// Construction is the only thing that needs the lock: `Transformer::new`
/// forces every parameter to be drawn before it returns, so a model that exists
/// is immune to later draws by anyone. Raw draws in the probes below take the
/// lock directly.
static RANDOM_SOURCE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquires the random-source lock, tolerating a previous test's panic.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    RANDOM_SOURCE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A fresh, uniquely named scratch directory inside the Cargo target directory.
fn scratch(name: &str) -> PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let directory =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("checkpoint-{name}-{unique}"));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).expect("scratch directory");
    directory
}

fn metadata(step: u64) -> CheckpointMetadata {
    CheckpointMetadata {
        format_version: CHECKPOINT_FORMAT_VERSION,
        training_semantics_version: llm_checkpoint::TRAINING_SEMANTICS_VERSION,
        created_unix_seconds: 1_700_000_000,
        global_step: step,
        tokens_seen: step * 512,
        data_cursor: DataCursor {
            epoch: 2,
            shard: 7,
            sequence: 1234,
        },
        config_hash: CONFIG.to_owned(),
        tokenizer_identity: TOKENIZER.to_owned(),
        dataset_manifest_hash: DATASET.to_owned(),
    }
}

fn expectation<'a>(
    tokenizer: &'a str,
    dataset: &'a str,
    config: Option<&'a str>,
) -> CheckpointExpectation<'a> {
    CheckpointExpectation {
        tokenizer_identity: tokenizer,
        dataset_manifest_hash: dataset,
        config_hash: config,
        training_semantics_version: llm_checkpoint::TRAINING_SEMANTICS_VERSION,
    }
}

fn tokens(device: &burn::tensor::Device) -> Tensor<2, Int> {
    Tensor::<2, Int>::from_data(TensorData::new(vec![1, 5, 9, 13, 17], [1, 5]), device)
}

/// Logits for a fixed input, which is what a restored model must reproduce.
fn logits(model: &Transformer, device: &burn::tensor::Device) -> Vec<f32> {
    model
        .forward(tokens(device), 0)
        .expect("shapes agree")
        .to_data()
        .try_to_vec::<f32>()
        .expect("f32")
}

/// Largest absolute difference between two logit vectors.
fn max_difference(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

/// A training batch taken from the committed overfit fixture, read as bytes.
fn training_batch(device: &burn::tensor::Device) -> TokenBatch {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("overfit-corpus.txt");
    let ids: Vec<i32> = std::fs::read(path)
        .expect("the fixture must be readable")
        .into_iter()
        .take(SEQUENCE + 1)
        .map(i32::from)
        .collect();

    let window = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, SEQUENCE + 1]), device);
    let (input, target) = next_token_pairs(window).expect("more than one token");

    TokenBatch {
        input,
        target,
        mask: all_positions(1, SEQUENCE, device),
    }
}

/// Writes a checkpoint and returns the model it stored.
fn write(root: &Path, name: &str, step: u64) -> Transformer {
    // `fresh_model` takes the random-source lock itself; taking it here as well
    // would deadlock, because the lock is not reentrant.
    let device = burn::tensor::Device::flex();
    let model = fresh_model(&device);
    checkpoint::save(root, name, &model, trainer().optimizer(), &metadata(step))
        .expect("the checkpoint writes");
    model
}

#[test]
fn the_device_seed_makes_two_constructions_identical() {
    // The premise every comparison in this file rests on. If seeding does not
    // actually make construction reproducible, two runs start from different
    // weights and every divergence measured here is meaningless.
    let device = burn::tensor::Device::flex();
    let first = fresh_model(&device);
    let second = fresh_model(&device);

    assert_eq!(
        logits(&first, &device),
        logits(&second, &device),
        "seeding the device did not make construction reproducible"
    );
}

/// One direct draw from the device's random source.
fn draw(device: &burn::tensor::Device) -> Vec<f32> {
    Tensor::<2>::random([8, 8], Distribution::Default, device)
        .to_data()
        .try_to_vec::<f32>()
        .expect("f32")
}

#[test]
fn every_random_distribution_the_project_uses_is_reproducible() {
    // Narrowing the cause: a direct draw is reproducible but the initializer is
    // not, so the question is which part of the draw differs. Testing each
    // distribution the project can use answers it in one run instead of one run
    // per guess.
    let _guard = exclusive();
    let device = burn::tensor::Device::flex();

    let cases: [(&str, Distribution); 4] = [
        ("Default", Distribution::Default),
        ("Normal(0, 1)", Distribution::Normal(0.0, 1.0)),
        (
            "Normal(0, weight_std)",
            Distribution::Normal(0.0, llm_model::init::WEIGHT_STD),
        ),
        ("Uniform(0, 1)", Distribution::Uniform(0.0, 1.0)),
    ];

    let mut failures = Vec::new();
    for (name, distribution) in cases {
        device.seed(SEED);
        let first = Tensor::<2>::random([8, 8], distribution, &device)
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");
        device.seed(SEED);
        let second = Tensor::<2>::random([8, 8], distribution, &device)
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");

        if first != second {
            failures.push(name);
        }
    }

    assert!(
        failures.is_empty(),
        "these distributions were not reproducible after re-seeding: {failures:?}"
    );
}

#[test]
fn seeding_is_reproducible_for_a_direct_random_draw() {
    // The probe that separates the two candidate causes of the failure above.
    //
    //   passes -> Flex seeding works, and the model's initializers are the ones
    //             bypassing the seeded random source;
    //   fails  -> seeding itself does not take effect on this backend.
    let _guard = exclusive();
    let device = burn::tensor::Device::flex();

    device.seed(SEED);
    let first = draw(&device);
    device.seed(SEED);
    let second = draw(&device);

    assert_eq!(
        first, second,
        "a direct random draw was not reproducible after re-seeding, so the fault \
         is in seeding rather than in the model's initializers"
    );
}

#[test]
fn seeding_is_reproducible_through_the_weight_initializer() {
    // The same probe, but through the initializer the model actually uses, with
    // the distribution the model actually draws from.
    let _guard = exclusive();
    let device = burn::tensor::Device::flex();
    let initializer = llm_model::init::weight_initializer();

    let mut drawn = Vec::new();
    for _ in 0..2 {
        device.seed(SEED);
        let tensor = initializer
            .init::<2, _>([8, 8], &device)
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");
        drawn.push(tensor);
    }

    assert_eq!(
        drawn[0], drawn[1],
        "the shared weight initializer did not draw the same values after re-seeding"
    );
}

#[test]
fn a_checkpoint_restores_the_parameters_exactly() {
    let root = scratch("round-trip");
    let device = burn::tensor::Device::flex();
    let saved = write(&root, "checkpoint-1", 5);
    let before = logits(&saved, &device);

    // A different model instance, loaded from disk, must behave identically.
    let blank = fresh_model(&device);
    let (restored, restored_metadata) = checkpoint::load_model(
        root.join("checkpoint-1"),
        blank,
        &expectation(TOKENIZER, DATASET, Some(CONFIG)),
    )
    .expect("the checkpoint loads");

    assert_eq!(restored_metadata, metadata(5));
    assert_eq!(before, logits(&restored, &device));
}

#[test]
fn the_envelope_carries_the_step_tokens_and_cursor() {
    let root = scratch("envelope");
    let _ = write(&root, "checkpoint-1", 11);

    let loaded = checkpoint::validate(root.join("checkpoint-1")).expect("valid envelope");
    assert_eq!(loaded.global_step, 11);
    assert_eq!(loaded.tokens_seen, 11 * 512);
    assert_eq!(
        loaded.data_cursor,
        DataCursor {
            epoch: 2,
            shard: 7,
            sequence: 1234
        }
    );
}

#[test]
fn publishing_a_checkpoint_moves_the_latest_pointer() {
    let root = scratch("latest");
    assert_eq!(checkpoint::latest(&root).expect("readable"), None);

    let _ = write(&root, "checkpoint-1", 1);
    assert_eq!(
        checkpoint::latest(&root).expect("readable").as_deref(),
        Some("checkpoint-1")
    );

    let _ = write(&root, "checkpoint-2", 2);
    assert_eq!(
        checkpoint::latest(&root).expect("readable").as_deref(),
        Some("checkpoint-2")
    );
    assert_eq!(
        checkpoint::latest_path(&root).expect("readable"),
        Some(root.join("checkpoint-2"))
    );
}

#[test]
fn a_failed_save_does_not_move_the_latest_pointer() {
    // The property that matters after a crash: whatever was published last is
    // still the newest thing a reader can find.
    let root = scratch("pointer-stability");
    let _ = write(&root, "checkpoint-1", 1);
    let published = checkpoint::latest(&root).expect("readable");

    let device = burn::tensor::Device::flex();
    let model = fresh_model(&device);
    assert!(matches!(
        checkpoint::save(
            &root,
            "checkpoint-1",
            &model,
            trainer().optimizer(),
            &metadata(9)
        ),
        Err(CheckpointError::AlreadyExists { .. })
    ));
    assert_eq!(checkpoint::latest(&root).expect("readable"), published);
}

#[test]
fn an_interrupted_write_leaves_no_referenced_checkpoint() {
    let root = scratch("interrupted");
    let _ = write(&root, "checkpoint-1", 1);

    // Simulate a crash mid-write: a temporary directory that was never renamed.
    let temporary = root.join(".tmp-checkpoint-2");
    std::fs::create_dir_all(&temporary).expect("scratch");
    std::fs::write(temporary.join(METADATA_FILE), "{}").expect("partial write");

    // The pointer still names the good checkpoint, and that checkpoint is sound.
    assert_eq!(
        checkpoint::latest(&root).expect("readable").as_deref(),
        Some("checkpoint-1")
    );
    assert!(checkpoint::validate(root.join("checkpoint-1")).is_ok());

    // A later save with the same name clears the stale directory and succeeds.
    let device = burn::tensor::Device::flex();
    let model = fresh_model(&device);
    checkpoint::save(
        &root,
        "checkpoint-2",
        &model,
        trainer().optimizer(),
        &metadata(2),
    )
    .expect("the retry writes");
    assert_eq!(
        checkpoint::latest(&root).expect("readable").as_deref(),
        Some("checkpoint-2")
    );
}

#[test]
fn an_envelope_from_another_format_version_is_rejected() {
    let root = scratch("version");
    let _ = write(&root, "checkpoint-1", 1);

    let path = root.join("checkpoint-1").join(METADATA_FILE);
    let mut envelope = checkpoint::validate(root.join("checkpoint-1")).expect("valid");
    envelope.format_version = CHECKPOINT_FORMAT_VERSION + 1;
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&envelope).expect("encodes"),
    )
    .expect("rewrite");

    assert!(matches!(
        checkpoint::validate(root.join("checkpoint-1")),
        Err(CheckpointError::UnsupportedVersion { .. })
    ));
}

#[test]
fn a_checkpoint_missing_its_parameters_is_rejected() {
    let root = scratch("incomplete");
    let _ = write(&root, "checkpoint-1", 1);
    std::fs::remove_file(root.join("checkpoint-1").join(MODEL_FILE)).expect("remove");

    assert!(matches!(
        checkpoint::validate(root.join("checkpoint-1")),
        Err(CheckpointError::Incomplete { .. })
    ));
}

#[test]
fn a_checkpoint_missing_its_optimizer_state_is_rejected() {
    // Without the optimizer state a resume restarts AdamW's moment estimates,
    // so a checkpoint that lacks it is not resumable and must not pass.
    let root = scratch("no-optimizer");
    let _ = write(&root, "checkpoint-1", 1);
    std::fs::remove_file(root.join("checkpoint-1").join(OPTIMIZER_FILE)).expect("remove");

    assert!(matches!(
        checkpoint::validate(root.join("checkpoint-1")),
        Err(CheckpointError::Incomplete { .. })
    ));
}

#[test]
fn a_corrupt_envelope_is_rejected() {
    let root = scratch("corrupt");
    let _ = write(&root, "checkpoint-1", 1);
    std::fs::write(
        root.join("checkpoint-1").join(METADATA_FILE),
        "{ this is not json",
    )
    .expect("rewrite");

    assert!(matches!(
        checkpoint::validate(root.join("checkpoint-1")),
        Err(CheckpointError::Decode { .. })
    ));
}

#[test]
fn loading_with_a_different_tokenizer_is_rejected() {
    let root = scratch("tokenizer");
    let _ = write(&root, "checkpoint-1", 1);
    let device = burn::tensor::Device::flex();
    let blank = fresh_model(&device);
    let other = "d".repeat(64);

    assert!(matches!(
        checkpoint::load_model(
            root.join("checkpoint-1"),
            blank,
            &expectation(&other, DATASET, Some(CONFIG))
        ),
        Err(CheckpointError::TokenizerMismatch { .. })
    ));
}

#[test]
fn loading_over_a_different_dataset_is_rejected() {
    let root = scratch("dataset");
    let _ = write(&root, "checkpoint-1", 1);
    let device = burn::tensor::Device::flex();
    let blank = fresh_model(&device);
    let other = "e".repeat(64);

    // Resuming over a different dataset would continue a run that never
    // happened, which is why this is a rejection and not a warning.
    assert!(matches!(
        checkpoint::load_model(
            root.join("checkpoint-1"),
            blank,
            &expectation(TOKENIZER, &other, Some(CONFIG))
        ),
        Err(CheckpointError::DatasetMismatch { .. })
    ));
}

#[test]
fn loading_with_a_different_configuration_is_rejected() {
    let root = scratch("config");
    let _ = write(&root, "checkpoint-1", 1);
    let device = burn::tensor::Device::flex();
    let blank = fresh_model(&device);
    let other = "f".repeat(64);

    assert!(matches!(
        checkpoint::load_model(
            root.join("checkpoint-1"),
            blank,
            &expectation(TOKENIZER, DATASET, Some(&other))
        ),
        Err(CheckpointError::ConfigMismatch { .. })
    ));
}

#[test]
fn an_unknown_configuration_hash_is_accepted() {
    // A caller who does not know the configuration hash is not asking for a
    // check, so an absent expectation must not fail the load.
    let root = scratch("config-absent");
    let _ = write(&root, "checkpoint-1", 1);
    let device = burn::tensor::Device::flex();
    let blank = fresh_model(&device);

    checkpoint::load_model(
        root.join("checkpoint-1"),
        blank,
        &expectation(TOKENIZER, DATASET, None),
    )
    .expect("no configuration check was requested");
}

#[test]
fn a_checkpoint_name_that_is_not_a_directory_name_is_rejected() {
    let root = scratch("name");
    let device = burn::tensor::Device::flex();
    let model = fresh_model(&device);

    for name in ["", "nested/checkpoint", "nested\\checkpoint"] {
        assert!(matches!(
            checkpoint::save(&root, name, &model, trainer().optimizer(), &metadata(1)),
            Err(CheckpointError::InvalidName { .. })
        ));
    }
}

#[test]
fn saving_twice_in_a_row_produces_two_independent_checkpoints() {
    // The first checkpoint must survive the second being written.
    let root = scratch("independent");
    let device = burn::tensor::Device::flex();
    let first = write(&root, "checkpoint-1", 1);
    let second = write(&root, "checkpoint-2", 2);

    let first_before = logits(&first, &device);
    let second_before = logits(&second, &device);

    for (name, expected) in [
        ("checkpoint-1", first_before),
        ("checkpoint-2", second_before),
    ] {
        let blank = fresh_model(&device);
        let (restored, _) = checkpoint::load_model(
            root.join(name),
            blank,
            &expectation(TOKENIZER, DATASET, Some(CONFIG)),
        )
        .expect("loads");
        assert_eq!(
            expected,
            logits(&restored, &device),
            "{name} restored wrong"
        );
    }
}

/// Seed every run starts from.
///
/// A model does not seed itself — the initializer policy puts that on the
/// composition root — so a test comparing two runs must seed before each
/// construction. Without this the runs begin from different weights and the
/// comparison measures nothing.
const SEED: u64 = 0x5EED_1234;

/// Builds a model at the test's fixed starting point.
fn fresh_model(device: &burn::tensor::Device) -> Transformer {
    let _guard = exclusive();
    llm_model::init::seed(device, SEED);
    Transformer::new(&tiny_config(), device).expect("valid config")
}

/// Trains `steps` steps from a fresh model and returns it.
fn trained(device: &burn::tensor::Device, steps: u64) -> Transformer {
    let batch = training_batch(device);
    let model = fresh_model(device);
    let mut trainer = trainer();
    let (model, _) = trainer
        .fit(model, std::slice::from_ref(&batch), steps)
        .expect("no step failed");
    model
}

/// Trains five steps and checkpoints the result, returning the checkpoint root.
fn five_steps_then_checkpoint(device: &burn::tensor::Device, name: &str) -> PathBuf {
    let root = scratch(name);
    let batch = training_batch(device);
    let model = fresh_model(device);
    let mut trainer = trainer();
    let (model, _) = trainer
        .fit(model, std::slice::from_ref(&batch), 5)
        .expect("no step failed");
    checkpoint::save(
        &root,
        "checkpoint-5",
        &model,
        trainer.optimizer(),
        &metadata(5),
    )
    .expect("the checkpoint writes");
    root
}

/// Loads a model from a checkpoint, ignoring its envelope.
fn load_model_only(root: &Path, device: &burn::tensor::Device) -> Transformer {
    let blank = fresh_model(device);
    let (model, _) = checkpoint::load_model(
        root.join("checkpoint-5"),
        blank,
        &expectation(TOKENIZER, DATASET, Some(CONFIG)),
    )
    .expect("the model loads");
    model
}

#[test]
fn a_resumed_run_continues_the_trajectory_of_an_uninterrupted_one() {
    // The acceptance criterion, stated as a comparison: ten steps in one go must
    // land where five, a checkpoint, a restart and five more land.
    let device = burn::tensor::Device::flex().autodiff();
    let batch = training_batch(&device);

    let uninterrupted = trained(&device, 10);
    let expected = logits(&uninterrupted, &device);

    let root = five_steps_then_checkpoint(&device, "resume");

    // A new process: a blank model and a fresh trainer, both rebuilt from disk.
    let restored = load_model_only(&root, &device);
    let mut resumed = trainer();
    let envelope = resumed
        .resume_from(root.join("checkpoint-5"))
        .expect("the optimizer state loads");
    assert_eq!(envelope.global_step, 5);
    assert_eq!(
        resumed.steps(),
        5,
        "the schedule position must be restored, or the learning rate restarts"
    );

    let (continued, _) = resumed
        .fit(restored, std::slice::from_ref(&batch), 5)
        .expect("no step failed");

    let difference = max_difference(&expected, &logits(&continued, &device));
    assert!(
        difference <= 1e-6,
        "a resumed run diverged from the uninterrupted one by {difference}"
    );
}

#[test]
fn resuming_with_the_in_memory_model_isolates_the_optimizer_restore() {
    // Diagnostic, and a boundary case worth keeping. The run below restores the
    // optimizer from disk but continues the *same model object* that was
    // checkpointed, rather than a reload of it. That separates two candidate
    // causes for the divergence above:
    //
    //   passes -> the optimizer state restores correctly, and the fault is that
    //             a reloaded model carries a different parameter identity from
    //             the one the optimizer state was keyed to;
    //   fails  -> the optimizer state itself is not being applied.
    let device = burn::tensor::Device::flex().autodiff();
    let batch = training_batch(&device);
    let expected = logits(&trained(&device, 10), &device);

    let root = scratch("resume-same-model");
    let model = fresh_model(&device);
    let mut first = trainer();
    let (model, _) = first
        .fit(model, std::slice::from_ref(&batch), 5)
        .expect("no step failed");
    checkpoint::save(
        &root,
        "checkpoint-5",
        &model,
        first.optimizer(),
        &metadata(5),
    )
    .expect("the checkpoint writes");

    let mut resumed = trainer();
    resumed
        .resume_from(root.join("checkpoint-5"))
        .expect("the optimizer state loads");

    // `model` here is the object that was saved, never written to disk and read
    // back. Only the optimizer crossed the file boundary.
    let (continued, _) = resumed
        .fit(model, std::slice::from_ref(&batch), 5)
        .expect("no step failed");

    let difference = max_difference(&expected, &logits(&continued, &device));
    assert!(
        difference <= 1e-6,
        "the optimizer state alone diverged by {difference}, so restoring it \
         from a file does not reproduce the state that was written"
    );
}

#[test]
fn resuming_without_the_optimizer_state_visibly_diverges() {
    // The comparison above would also pass if restoring the optimizer state did
    // nothing and AdamW happened to arrive at the same place. This rules that
    // out: resuming with a fresh optimizer must land somewhere else, which is
    // what makes the equality above evidence rather than coincidence.
    let device = burn::tensor::Device::flex().autodiff();
    let batch = training_batch(&device);

    let uninterrupted = trained(&device, 10);
    let expected = logits(&uninterrupted, &device);

    let root = five_steps_then_checkpoint(&device, "resume-without-optimizer");
    let restored = load_model_only(&root, &device);

    // Note the omission: the trainer is fresh and is never told to resume.
    let mut forgetful = trainer();
    let (continued, _) = forgetful
        .fit(restored, std::slice::from_ref(&batch), 5)
        .expect("no step failed");

    let difference = max_difference(&expected, &logits(&continued, &device));
    assert!(
        difference > 1e-6,
        "losing the optimizer state changed nothing ({difference}), so the \
         resume test above would prove nothing"
    );
}
