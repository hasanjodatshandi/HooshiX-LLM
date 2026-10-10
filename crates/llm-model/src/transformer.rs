//! The decoder-only Transformer.
//!
//! Assembly of the primitives, and the only place that knows the whole model.
//! Construction validates the architecture before any tensor is allocated, and
//! refuses a configuration this stage does not implement rather than
//! approximating it.

use burn::module::Module;
use burn::tensor::{Device, Int, Tensor};
use llm_core::ModelConfig;

use crate::block::TransformerBlock;
use crate::embedding::TokenEmbedding;
use crate::error::ModelError;
use crate::kv_cache::KvCache;
use crate::rms_norm::RmsNorm;
use crate::rope::Rope;

/// A decoder-only causal language model.
#[derive(Module, Debug)]
pub struct Transformer {
    embedding: TokenEmbedding,
    blocks: Vec<TransformerBlock>,
    final_norm: RmsNorm,
}

impl Transformer {
    /// Builds the model described by `config`.
    ///
    /// The configuration is validated first, so an incoherent architecture fails
    /// before a parameter is allocated.
    pub fn new(config: &ModelConfig, device: &Device) -> Result<Self, ModelError> {
        config
            .validate()
            .map_err(|error| ModelError::InvalidConfig {
                reason: error.to_string(),
            })?;

        // Stage 1 is plain multi-head attention. Grouped-query attention is an
        // explicit Stage-2 change, so a grouped configuration is refused here
        // instead of being silently widened to full multi-head.
        if config.num_kv_heads != config.num_heads {
            return Err(ModelError::InvalidConfig {
                reason: format!(
                    "Stage 1 implements multi-head attention only, so num_kv_heads must equal \
                     num_heads; got {} query heads and {} key/value heads",
                    config.num_heads, config.num_kv_heads
                ),
            });
        }

        let head_dim = config.head_dim().ok_or_else(|| ModelError::InvalidConfig {
            reason: "hidden size must be divisible by the number of heads".to_owned(),
        })?;
        let rope = Rope::new(head_dim, config.rope_theta)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(config, head_dim, &rope, device));
        }

        Ok(Self {
            embedding: TokenEmbedding::new(config.vocab_size, config.hidden_size, device),
            blocks,
            final_norm: RmsNorm::new(config.hidden_size, config.rms_norm_eps, device),
        })
        .inspect(|model| {
            // Burn draws parameters lazily. Forcing the draw here means the
            // seed alone decides the weights, instead of the seed plus whatever
            // else happened to read the device's random source first.
            crate::init::materialize(model);
        })
    }

    /// Number of decoder blocks.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// The shared embedding and head matrix, exposed for the tying test.
    #[must_use]
    pub fn embedding_weight(&self) -> Tensor<2> {
        self.embedding.shared_weight()
    }

    /// Runs the model over `[batch, sequence]` token ids and returns
    /// `[batch, sequence, vocab]` logits.
    ///
    /// `position_offset` is the absolute position of the first token, so a
    /// cached single-token step lands at its true position.
    pub fn forward(
        &self,
        tokens: Tensor<2, Int>,
        position_offset: usize,
    ) -> Result<Tensor<3>, ModelError> {
        let mut hidden = self.embedding.embed(tokens)?;
        for block in &self.blocks {
            hidden = block.forward(hidden, position_offset)?;
        }
        let normalized = self.final_norm.forward(hidden);
        self.embedding.project(normalized)
    }

    /// Runs the model over `[batch, sequence]` token ids, extending `cache`.
    ///
    /// This is **prefill** when the cache is empty and a **decode step** when it
    /// already holds the prompt. `position_offset` must equal the number of
    /// positions the cache holds; the two are the same quantity, and disagreeing
    /// means the caller has lost track of where the sequence is.
    ///
    /// The result is identical to [`forward`](Self::forward) over the whole
    /// sequence so far — that equivalence is what makes the cache safe, and it is
    /// tested rather than assumed.
    pub fn forward_cached(
        &self,
        tokens: Tensor<2, Int>,
        cache: &mut KvCache,
        position_offset: usize,
    ) -> Result<Tensor<3>, ModelError> {
        let mut hidden = self.embedding.embed(tokens)?;
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block.forward_cached(hidden, cache.layer(index)?, position_offset)?;
        }
        let normalized = self.final_norm.forward(hidden);
        self.embedding.project(normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::Transformer;
    use crate::testing::reference_device;
    use burn::tensor::{Int, Tensor, TensorData};
    use llm_core::ModelConfig;

    /// Small architecture that keeps the tests fast while exercising every rule.
    fn small() -> ModelConfig {
        ModelConfig {
            vocab_size: 64,
            hidden_size: 16,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            ffn_intermediate_size: 32,
            max_sequence_length: 32,
            ..ModelConfig::default()
        }
    }

    fn tokens(
        values: Vec<i32>,
        shape: [usize; 2],
        device: &burn::tensor::Device,
    ) -> Tensor<2, Int> {
        Tensor::<2, Int>::from_data(TensorData::new(values, shape), device)
    }

    #[test]
    fn the_forward_pass_produces_batch_sequence_vocabulary() {
        let device = reference_device();
        let config = small();
        let model = Transformer::new(&config, &device).expect("valid config");

        let logits = model
            .forward(tokens(vec![1, 2, 3, 4, 5, 6], [2, 3], &device), 0)
            .expect("shapes agree");

        assert_eq!(logits.dims(), [2, 3, config.vocab_size]);
        assert_eq!(model.num_layers(), config.num_layers);
    }

    #[test]
    fn the_parameter_count_matches_the_configuration_accounting() {
        // The configuration predicts a count from its dimensions; the built
        // model counts what it actually allocated. They must agree, or the
        // Stage-1 parameter budget is fiction.
        use burn::module::Module as _;

        let device = reference_device();
        let config = small();
        let model = Transformer::new(&config, &device).expect("valid config");

        let predicted = config.parameter_count().expect("fits in u64");
        assert_eq!(model.num_params() as u64, predicted);
    }

    #[test]
    fn the_language_model_head_is_tied_to_the_embedding() {
        use burn::module::Module as _;

        let device = reference_device();
        let config = small();
        let model = Transformer::new(&config, &device).expect("valid config");

        // Exactly one matrix of `vocab_size * hidden_size` exists, and the
        // head reads it: an untied model would carry a second one.
        let shared = model.embedding_weight();
        assert_eq!(shared.dims(), [config.vocab_size, config.hidden_size]);

        let untied = config.parameter_count().expect("fits")
            + (config.vocab_size as u64 * config.hidden_size as u64);
        assert_ne!(model.num_params() as u64, untied);
    }

    #[test]
    fn the_model_is_causal_end_to_end() {
        let device = reference_device();
        let model = Transformer::new(&small(), &device).expect("valid config");

        let base = tokens(vec![1, 2, 3, 4], [1, 4], &device);
        let first_before = model
            .forward(base.clone(), 0)
            .expect("shapes agree")
            .narrow(1, 0, 1);

        let perturbed = tokens(vec![1, 2, 3, 9], [1, 4], &device);
        let first_after = model
            .forward(perturbed, 0)
            .expect("shapes agree")
            .narrow(1, 0, 1);

        let before = first_before.to_data().try_to_vec::<f32>().expect("f32");
        let after = first_after.to_data().try_to_vec::<f32>().expect("f32");
        for (left, right) in before.iter().zip(after.iter()) {
            assert!(
                (left - right).abs() < 1e-5,
                "the first position responded to a later token: {left} vs {right}"
            );
        }
    }

    #[test]
    fn a_grouped_query_configuration_is_refused() {
        let device = reference_device();
        let config = ModelConfig {
            num_kv_heads: 2,
            ..small()
        };

        let error = Transformer::new(&config, &device).expect_err("Stage 1 is multi-head only");
        assert!(
            error.to_string().contains("multi-head attention only"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn an_invalid_configuration_fails_before_construction() {
        let device = reference_device();
        let config = ModelConfig {
            hidden_size: 15,
            ..small()
        };

        assert!(Transformer::new(&config, &device).is_err());
    }

    #[test]
    fn the_output_is_finite_for_a_freshly_initialized_model() {
        let device = reference_device();
        let model = Transformer::new(&small(), &device).expect("valid config");

        let logits = model
            .forward(tokens(vec![7, 11, 13], [1, 3], &device), 0)
            .expect("shapes agree");
        let values = logits.to_data().try_to_vec::<f32>().expect("f32");

        assert!(
            values.iter().all(|value| value.is_finite()),
            "a fresh model must produce finite logits"
        );
        assert!(
            values.iter().any(|value| value.abs() > 1e-6),
            "a fresh model must produce non-trivial logits"
        );
    }

    #[test]
    fn a_position_offset_moves_the_result() {
        let device = reference_device();
        let model = Transformer::new(&small(), &device).expect("valid config");
        let input = tokens(vec![3, 5], [1, 2], &device);

        let from_zero = model
            .forward(input.clone(), 0)
            .expect("shapes agree")
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");
        let from_five = model
            .forward(input, 5)
            .expect("shapes agree")
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");

        let moved = from_zero
            .iter()
            .zip(from_five.iter())
            .any(|(left, right)| (left - right).abs() > 1e-6);
        assert!(moved, "rotary positions must depend on the offset");
    }

    #[test]
    fn a_backward_pass_reaches_the_shared_embedding_with_a_non_zero_gradient() {
        let device = reference_device().autodiff();
        let model = Transformer::new(&small(), &device).expect("valid config");
        let parameter = model.embedding.shared_parameter();

        let objective = model
            .forward(tokens(vec![1, 2, 3], [1, 3], &device), 0)
            .expect("shapes agree")
            .sum();
        let gradients = objective.backward();

        let gradient = parameter
            .grad(&gradients)
            .expect("the shared embedding participates in the graph");
        crate::grad::ensure_finite(&gradient, crate::error::NumericalQuantity::Gradients)
            .expect("the gradient is finite");

        let values = gradient.to_data().try_to_vec::<f32>().expect("f32");
        assert!(
            values.iter().any(|value| value.abs() > 1e-12),
            "the gradient of the shared embedding must not be entirely zero"
        );
    }

    /// Logits as a plain vector, for comparison.
    fn logits_of(tensor: &Tensor<3>) -> Vec<f32> {
        tensor.to_data().try_to_vec::<f32>().expect("f32")
    }

    /// Largest absolute difference between two logit vectors.
    fn divergence(left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len(), "compared tensors differ in size");
        left.iter()
            .zip(right.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn cached_decoding_reproduces_a_full_forward_pass() {
        // The property that makes a key/value cache safe. A wrong mask, a
        // mis-rotated position or a cache that replaced instead of extended
        // would all show up here as a divergence.
        use crate::kv_cache::KvCache;

        let device = reference_device();
        let config = small();
        let model = Transformer::new(&config, &device).expect("valid config");

        let prompt = [1, 2, 3];
        let continuation = [4, 5, 6, 7];
        let sequence: Vec<i32> = prompt.iter().chain(continuation.iter()).copied().collect();

        let full = model
            .forward(tokens(sequence.clone(), [1, sequence.len()], &device), 0)
            .expect("shapes agree");

        // Prefill the prompt, then decode one token at a time.
        let mut cache = KvCache::new(config.num_layers);
        let prefill = model
            .forward_cached(
                tokens(prompt.to_vec(), [1, prompt.len()], &device),
                &mut cache,
                0,
            )
            .expect("shapes agree");
        assert_eq!(cache.len(), prompt.len());

        let prefill_expected = full.clone().narrow(1, 0, prompt.len());
        let difference = divergence(&logits_of(&prefill), &logits_of(&prefill_expected));
        assert!(
            difference < 1e-5,
            "prefill diverged from the full pass by {difference}"
        );

        for (step, token) in continuation.iter().enumerate() {
            let position = prompt.len() + step;
            let latest = model
                .forward_cached(tokens(vec![*token], [1, 1], &device), &mut cache, position)
                .expect("shapes agree");
            assert_eq!(cache.len(), position + 1);

            // Every step must also agree with the full pass at that position,
            // not merely the last one: an error that cancels at the end would
            // otherwise pass.
            let expected = full.clone().narrow(1, position, 1);
            let difference = divergence(&logits_of(&latest), &logits_of(&expected));
            assert!(
                difference < 1e-5,
                "decode step at position {position} diverged by {difference}"
            );
        }
    }

    #[test]
    fn a_cached_run_extends_the_cache_by_exactly_the_tokens_it_read() {
        use crate::kv_cache::KvCache;

        let device = reference_device();
        let config = small();
        let model = Transformer::new(&config, &device).expect("valid config");
        let mut cache = KvCache::new(config.num_layers);

        assert!(cache.is_empty());

        let _ = model
            .forward_cached(tokens(vec![1, 2], [1, 2], &device), &mut cache, 0)
            .expect("shapes agree");
        assert_eq!(cache.len(), 2);

        let _ = model
            .forward_cached(tokens(vec![3], [1, 1], &device), &mut cache, 2)
            .expect("shapes agree");
        assert_eq!(cache.len(), 3);

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.num_layers(), config.num_layers);
    }

    #[test]
    fn a_cache_built_for_another_depth_is_reported() {
        use crate::kv_cache::KvCache;

        let device = reference_device();
        let config = small();
        let model = Transformer::new(&config, &device).expect("valid config");

        // One layer short: the model needs an entry the cache does not have.
        let mut too_shallow = KvCache::new(config.num_layers - 1);
        assert!(matches!(
            model.forward_cached(tokens(vec![1], [1, 1], &device), &mut too_shallow, 0),
            Err(crate::error::ModelError::CacheLayer { .. })
        ));
    }
}
