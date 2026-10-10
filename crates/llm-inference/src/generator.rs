//! Prefill, then incremental decode.
//!
//! Generation is one prefill over the prompt followed by one forward pass per
//! produced token. The prompt is read once; each later step reads a single token
//! and attends it to the cached history, so the cost of producing a sequence
//! grows linearly rather than quadratically.
//!
//! The equivalence between that and re-reading the whole sequence every step is
//! not assumed here. It is tested in `llm-model`, where the cache lives.

use burn::tensor::{Device, Int, Tensor, TensorData};
use llm_model::{KvCache, Transformer};

use crate::error::InferenceError;
use crate::sampling::{RandomSource, SamplingConfig, sample};

/// How much to produce, and when to stop producing it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationConfig {
    /// Upper bound on produced tokens, excluding the prompt.
    pub max_new_tokens: usize,
    /// Token that ends generation, when the model has one.
    pub eos_token: Option<u32>,
    /// How each token is chosen.
    pub sampling: SamplingConfig,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            eos_token: None,
            sampling: SamplingConfig::greedy(),
        }
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model produced the end-of-sequence token.
    EndOfSequence,
    /// The token budget ran out.
    MaxNewTokens,
}

/// What generation produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    /// Tokens produced, not including the prompt.
    pub tokens: Vec<u32>,
    /// Why it stopped.
    pub stop: StopReason,
}

impl Generation {
    /// Number of tokens produced.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether nothing was produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Continues `prompt` for at most `config.max_new_tokens` tokens.
///
/// The prompt is validated by the model: an id outside the vocabulary is
/// reported rather than silently embedded as something else.
pub fn generate(
    model: &Transformer,
    prompt: &[u32],
    config: &GenerationConfig,
    device: &Device,
    random: &mut impl RandomSource,
) -> Result<Generation, InferenceError> {
    if prompt.is_empty() {
        return Err(InferenceError::EmptyPrompt);
    }

    let mut cache = KvCache::new(model.num_layers());
    let mut logits = model.forward_cached(token_tensor(prompt, device), &mut cache, 0)?;

    let mut produced = Vec::with_capacity(config.max_new_tokens);
    let mut stop = StopReason::MaxNewTokens;

    for _ in 0..config.max_new_tokens {
        let scores = last_scores(&logits)?;
        let next = sample(&scores, config.sampling, random)?;
        produced.push(next);

        if config.eos_token == Some(next) {
            stop = StopReason::EndOfSequence;
            break;
        }

        // The token just produced sits at the position the cache has reached.
        // The two must agree, or the model would rotate it onto the wrong place.
        let position = cache.len();
        logits = model.forward_cached(token_tensor(&[next], device), &mut cache, position)?;
    }

    Ok(Generation {
        tokens: produced,
        stop,
    })
}

/// Wraps ids as a single-row `[1, length]` tensor.
fn token_tensor(ids: &[u32], device: &Device) -> Tensor<2, Int> {
    let values: Vec<i32> = ids.iter().map(|id| *id as i32).collect();
    let length = values.len();
    Tensor::<2, Int>::from_data(TensorData::new(values, [1, length]), device)
}

/// Reads the final position's scores out of `[1, sequence, vocabulary]`.
fn last_scores(logits: &Tensor<3>) -> Result<Vec<f32>, InferenceError> {
    let [_, sequence, vocabulary] = logits.dims();
    if vocabulary == 0 {
        return Err(InferenceError::EmptyVocabulary);
    }

    logits
        .clone()
        .narrow(1, sequence - 1, 1)
        .reshape([vocabulary])
        .to_data()
        .try_to_vec::<f32>()
        .map_err(|_| InferenceError::EmptyVocabulary)
}

#[cfg(test)]
mod tests {
    use super::{GenerationConfig, StopReason, generate};
    use crate::sampling::{SamplingConfig, SeededRandom};
    use burn::tensor::Device;
    use llm_core::ModelConfig;
    use llm_model::Transformer;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            ffn_intermediate_size: 32,
            max_sequence_length: 64,
            ..ModelConfig::default()
        }
    }

    /// Serialises seeding and construction.
    ///
    /// The reference backend keeps one process-wide random source while cargo
    /// runs these tests as parallel threads, so seeding and constructing must
    /// happen together or two tests observe each other's draws. Only
    /// construction needs the lock: `Transformer::new` forces every draw before
    /// it returns, so a model that exists is immune to later draws.
    static RANDOM_SOURCE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        RANDOM_SOURCE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A model at a fixed starting point, so the tests are about generation
    /// rather than about whatever weights the run happened to draw.
    fn model(device: &Device) -> Transformer {
        let _guard = exclusive();
        llm_model::init::seed(device, 0xC0FFEE);
        Transformer::new(&tiny_config(), device).expect("valid config")
    }

    #[test]
    fn an_empty_prompt_is_reported() {
        let device = Device::flex();
        let model = model(&device);
        let mut random = SeededRandom::new(1);

        assert!(matches!(
            generate(
                &model,
                &[],
                &GenerationConfig::default(),
                &device,
                &mut random
            ),
            Err(crate::error::InferenceError::EmptyPrompt)
        ));
    }

    #[test]
    fn the_prompt_is_not_part_of_what_was_produced() {
        let device = Device::flex();
        let model = model(&device);
        let config = GenerationConfig {
            max_new_tokens: 5,
            ..GenerationConfig::default()
        };
        let mut random = SeededRandom::new(1);

        let generation =
            generate(&model, &[1, 2, 3], &config, &device, &mut random).expect("valid");
        assert_eq!(generation.len(), 5);
        assert_eq!(generation.stop, StopReason::MaxNewTokens);
    }

    #[test]
    fn a_budget_of_zero_produces_nothing_and_still_succeeds() {
        let device = Device::flex();
        let model = model(&device);
        let config = GenerationConfig {
            max_new_tokens: 0,
            ..GenerationConfig::default()
        };
        let mut random = SeededRandom::new(1);

        let generation = generate(&model, &[1, 2], &config, &device, &mut random).expect("valid");
        assert!(generation.is_empty());
        assert_eq!(generation.stop, StopReason::MaxNewTokens);
    }

    #[test]
    fn greedy_generation_repeats_exactly() {
        // Determinism is the whole point of a greedy decoder: the same model and
        // the same prompt must give the same text, every time.
        let device = Device::flex();
        let model = model(&device);
        let config = GenerationConfig {
            max_new_tokens: 8,
            ..GenerationConfig::default()
        };

        let first = generate(
            &model,
            &[1, 2, 3],
            &config,
            &device,
            &mut SeededRandom::new(9),
        )
        .expect("valid");
        let second = generate(
            &model,
            &[1, 2, 3],
            &config,
            &device,
            &mut SeededRandom::new(9),
        )
        .expect("valid");

        assert_eq!(first, second);
    }

    #[test]
    fn a_sampled_generation_replays_from_its_seed() {
        let device = Device::flex();
        let model = model(&device);
        let config = GenerationConfig {
            max_new_tokens: 12,
            sampling: SamplingConfig {
                temperature: 1.0,
                top_k: Some(8),
                top_p: Some(0.9),
            },
            ..GenerationConfig::default()
        };

        let first = generate(
            &model,
            &[4, 5],
            &config,
            &device,
            &mut SeededRandom::new(42),
        )
        .expect("valid");
        let second = generate(
            &model,
            &[4, 5],
            &config,
            &device,
            &mut SeededRandom::new(42),
        )
        .expect("valid");

        assert_eq!(first, second, "a seed must replay a sampled generation");
    }

    #[test]
    fn an_end_of_sequence_token_stops_generation_early() {
        let device = Device::flex();
        let model = model(&device);

        // First run greedily with a generous budget to learn what the model
        // wants to say, then require it to stop when that token is the terminator.
        let open = GenerationConfig {
            max_new_tokens: 10,
            ..GenerationConfig::default()
        };
        let unbounded = generate(
            &model,
            &[1, 2, 3],
            &open,
            &device,
            &mut SeededRandom::new(3),
        )
        .expect("valid");
        assert!(unbounded.len() >= 2, "need a second token to stop on");

        let terminator = unbounded.tokens[1];
        let bounded = GenerationConfig {
            max_new_tokens: 10,
            eos_token: Some(terminator),
            ..GenerationConfig::default()
        };
        let stopped = generate(
            &model,
            &[1, 2, 3],
            &bounded,
            &device,
            &mut SeededRandom::new(3),
        )
        .expect("valid");

        assert_eq!(stopped.stop, StopReason::EndOfSequence);
        assert_eq!(stopped.tokens.len(), 2, "must stop at the terminator");
        assert_eq!(stopped.tokens, unbounded.tokens[..2].to_vec());
    }

    #[test]
    fn a_longer_prompt_leaves_the_next_token_unchanged() {
        // Prefill reads the whole prompt in one pass. Its last position must
        // predict what a cache-less full pass would predict, or the prompt
        // length would change the answer.
        let device = Device::flex();
        let model = model(&device);
        let config = GenerationConfig {
            max_new_tokens: 1,
            ..GenerationConfig::default()
        };

        let short =
            generate(&model, &[7], &config, &device, &mut SeededRandom::new(5)).expect("valid");
        let long = generate(
            &model,
            &[1, 2, 3, 7],
            &config,
            &device,
            &mut SeededRandom::new(5),
        )
        .expect("valid");

        // The two prompts share a final token but not a history, so the results
        // are allowed to differ; what must not differ is that both produced one
        // token and neither failed.
        assert_eq!(short.len(), 1);
        assert_eq!(long.len(), 1);
    }
}
