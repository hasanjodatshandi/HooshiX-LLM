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
fn shipped_configs_round_trip_through_toml() {
    for name in ["stage1-110m.toml", "stage1-smoke.toml"] {
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
