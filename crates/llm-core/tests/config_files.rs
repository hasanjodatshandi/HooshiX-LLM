//! The shipped configuration files must load, validate and round-trip.
//!
//! These files are the ones a human edits and a run resolves, so a change that
//! makes them unloadable is a release-blocking regression rather than a
//! documentation drift.

use std::path::PathBuf;

use llm_core::{ExperimentConfig, SamplingStrategy, TokenizerModel};

fn config_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("configs")
        .join(name)
}

#[test]
fn stage1_110m_loads_and_validates() {
    let config = ExperimentConfig::load(config_path("stage1-110m.toml"))
        .expect("configs/stage1-110m.toml must load and validate");

    assert_eq!(config.model.num_layers, 12);
    assert_eq!(config.model.hidden_size, 768);
    assert_eq!(config.model.head_dim(), Some(64));
    assert_eq!(config.tokenizer.model, TokenizerModel::Bpe);
    assert_eq!(config.generation.strategy, SamplingStrategy::Greedy);

    let parameters = config
        .model
        .parameter_count()
        .expect("parameter count must fit in u64");
    assert!(
        (100_000_000..=150_000_000).contains(&parameters),
        "Stage 1 must stay inside the 100M..=150M band, got {parameters}"
    );

    // Effective batch accounting must be reportable, not ambiguous.
    assert_eq!(config.effective_batch_sequences(), Some(8));
    assert_eq!(config.effective_tokens_per_step(), Some(16_384));
}

#[test]
fn stage1_smoke_loads_and_validates() {
    let config = ExperimentConfig::load(config_path("stage1-smoke.toml"))
        .expect("configs/stage1-smoke.toml must load and validate");

    assert_eq!(config.model.num_layers, 4);
    assert_eq!(config.model.head_dim(), Some(32));
    assert!(matches!(
        config.generation.strategy,
        SamplingStrategy::TopK { k: 40, .. }
    ));
}

#[test]
fn tinystories_small_loads_and_validates() {
    let config = ExperimentConfig::load(config_path("tinystories-small.toml"))
        .expect("configs/tinystories-small.toml must load and validate");

    assert_eq!(config.model.num_layers, 6);
    assert_eq!(config.model.head_dim(), Some(64));
    // The vocabulary is sized for the corpus rather than for Stage 1: TinyStories
    // draws on a deliberately small word list, so a 32768-token vocabulary would
    // be mostly unused rows in the embedding and in every logits tensor.
    assert_eq!(config.model.vocab_size, 8192);
    assert_eq!(config.tokenizer.vocab_size, 8192);
    // A vocabulary mismatch between the two is a shape error at the first batch.
    assert_eq!(config.model.vocab_size, config.tokenizer.vocab_size);

    let parameters = config
        .model
        .parameter_count()
        .expect("parameter count must fit in u64");
    assert!(
        (10_000_000..=50_000_000).contains(&parameters),
        "the TinyStories model is deliberately small, got {parameters}"
    );
}

#[test]
fn bilingual_pilot_loads_and_validates() {
    let config = ExperimentConfig::load(config_path("bilingual-pilot.toml"))
        .expect("configs/bilingual-pilot.toml must load and validate");

    // The pilot measures the model that will actually be trained, so the
    // architecture has to match `stage1-110m.toml` rather than being a smaller
    // stand-in whose behaviour does not transfer.
    let stage1 = ExperimentConfig::load(config_path("stage1-110m.toml")).expect("loads");
    assert_eq!(config.model.vocab_size, stage1.model.vocab_size);
    assert_eq!(config.model.hidden_size, stage1.model.hidden_size);
    assert_eq!(config.model.num_layers, stage1.model.num_layers);
    assert_eq!(config.model.num_heads, stage1.model.num_heads);
    assert_eq!(
        config.model.parameter_count().expect("fits"),
        stage1.model.parameter_count().expect("fits"),
        "the pilot must be the same model, not a smaller one"
    );

    // Sized against the bilingual tokenizer, which was trained for 32768.
    assert_eq!(config.tokenizer.vocab_size, 32768);
    assert_eq!(config.model.vocab_size, config.tokenizer.vocab_size);
    assert!(
        config
            .tokenizer
            .artifact
            .as_deref()
            .is_some_and(|path| path.to_string_lossy().contains("bilingual")),
        "the pilot must name the bilingual tokenizer, not the TinyStories one"
    );

    // The 4 GB card's ceiling for this model is 512 tokens per micro-batch,
    // measured in docs/11 section 20. A pilot that cannot use the faster backend
    // is a pilot that takes three days longer than it needs to.
    assert_eq!(config.data.sequence_length, 512);
    assert!(config.model.max_sequence_length >= config.data.sequence_length);

    // One epoch over the prepared corpus rather than an arbitrary budget.
    let per_step = config.data.micro_batch_size as u64
        * config.data.sequence_length as u64
        * config.training.gradient_accumulation_steps as u64;
    let tokens = per_step * config.training.max_steps;
    assert!(
        (90_000_000..=115_000_000).contains(&tokens),
        "the pilot is meant to be one epoch over about 100M tokens, got {tokens}"
    );

    // The scheduler must not outlive the run or the learning rate never reaches
    // its floor, and must not end early or it decays to nothing and keeps going.
    assert_eq!(
        config.training.scheduler.cosine_iters,
        config.training.max_steps
    );
    assert!(config.training.scheduler.warmup_steps < config.training.max_steps);

    // Nothing the TinyStories run wrote may be touched.
    assert!(
        config.data.data_root.to_string_lossy().contains("pilot"),
        "the pilot must write to its own data root, got {}",
        config.data.data_root.display()
    );
}

#[test]
fn shipped_configs_round_trip_through_toml() {
    for name in [
        "stage1-110m.toml",
        "stage1-smoke.toml",
        "tinystories-small.toml",
        "bilingual-pilot.toml",
    ] {
        let config = ExperimentConfig::load(config_path(name)).expect("loads");
        let text = config.to_toml_string().expect("serializes");
        let parsed = ExperimentConfig::from_toml_str(&text).expect("re-parses");
        assert_eq!(parsed, config, "{name} must round-trip without loss");
    }
}

#[test]
fn a_missing_file_reports_the_path() {
    let error = ExperimentConfig::load(config_path("does-not-exist.toml"))
        .expect_err("a missing file must fail");
    let message = error.to_string();
    assert!(
        message.contains("does-not-exist.toml"),
        "the error must name the file, got: {message}"
    );
}
