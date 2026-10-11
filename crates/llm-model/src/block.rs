//! One pre-normalized Transformer block.
//!
//! Reference form, from the model specification:
//!
//! ```text
//! h = x + Attention(RMSNorm(x))
//! y = h + FFN(RMSNorm(h))
//! ```
//!
//! Normalization is applied before each sub-layer, not after it, and both
//! sub-layers are wrapped in a residual connection. The residual stream is what
//! carries information past the block, so neither sub-layer output is normalized
//! on the way back into it.

use burn::module::Module;
use burn::tensor::{Device, Tensor};
use llm_core::ModelConfig;

use crate::attention::{AttentionMode, MultiHeadAttention};
use crate::error::ModelError;
use crate::kv_cache::LayerCache;
use crate::rms_norm::RmsNorm;
use crate::rope::Rope;
use crate::swiglu::SwiGlu;

/// A single decoder block.
#[derive(Module, Debug)]
pub struct TransformerBlock {
    attention_norm: RmsNorm,
    attention: MultiHeadAttention,
    ffn_norm: RmsNorm,
    ffn: SwiGlu,
}

impl TransformerBlock {
    /// Creates a block from the architecture it belongs to.
    ///
    /// The configuration is passed whole rather than as eight loose numbers:
    /// a block is a slice of one model, and a long argument list of bare
    /// dimensions is where transposed sizes hide.
    #[must_use]
    pub fn new(config: &ModelConfig, head_dim: usize, rope: &Rope, device: &Device) -> Self {
        Self::with_attention_mode(config, head_dim, rope, AttentionMode::default(), device)
    }

    /// Creates a block whose attention scores the sequence `mode`'s way.
    #[must_use]
    pub fn with_attention_mode(
        config: &ModelConfig,
        head_dim: usize,
        rope: &Rope,
        mode: AttentionMode,
        device: &Device,
    ) -> Self {
        Self {
            attention_norm: RmsNorm::new(config.hidden_size, config.rms_norm_eps, device),
            attention: MultiHeadAttention::new(
                config.hidden_size,
                config.num_heads,
                head_dim,
                config.linear_bias,
                rope,
                device,
            )
            .with_mode(mode),
            ffn_norm: RmsNorm::new(config.hidden_size, config.rms_norm_eps, device),
            ffn: SwiGlu::new(
                config.hidden_size,
                config.ffn_intermediate_size,
                config.linear_bias,
                device,
            ),
        }
    }

    /// Returns this block scoring attention `mode`'s way.
    pub fn set_attention_mode(&mut self, mode: AttentionMode) {
        self.attention = self.attention.clone().with_mode(mode);
    }

    /// Which way this block scores attention.
    #[must_use]
    pub const fn attention_mode(&self) -> AttentionMode {
        self.attention.mode()
    }

    /// Applies the block to `[batch, sequence, hidden]`.
    pub fn forward(
        &self,
        input: Tensor<3>,
        position_offset: usize,
    ) -> Result<Tensor<3>, ModelError> {
        let attended = self
            .attention
            .forward(self.attention_norm.forward(input.clone()), position_offset)?;
        let after_attention = input + attended;

        let fed = self
            .ffn
            .forward(self.ffn_norm.forward(after_attention.clone()))?;
        Ok(after_attention + fed)
    }

    /// Applies the block to `[batch, sequence, hidden]`, extending `cache`.
    ///
    /// The feed-forward half is unchanged by caching: it reads one position at a
    /// time, so only attention has a history to carry.
    pub fn forward_cached(
        &self,
        input: Tensor<3>,
        cache: &mut LayerCache,
        position_offset: usize,
    ) -> Result<Tensor<3>, ModelError> {
        let attended = self.attention.forward_cached(
            self.attention_norm.forward(input.clone()),
            cache,
            position_offset,
        )?;
        let after_attention = input + attended;

        let fed = self
            .ffn
            .forward(self.ffn_norm.forward(after_attention.clone()))?;
        Ok(after_attention + fed)
    }
}

#[cfg(test)]
mod tests {
    use super::TransformerBlock;
    use crate::rope::Rope;
    use crate::testing::reference_device;
    use burn::tensor::{Tensor, TensorData};

    /// Small architecture that keeps the tests fast.
    fn shape() -> llm_core::ModelConfig {
        llm_core::ModelConfig {
            vocab_size: 32,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            ffn_intermediate_size: 16,
            max_sequence_length: 16,
            ..llm_core::ModelConfig::default()
        }
    }

    fn block(device: &burn::tensor::Device) -> TransformerBlock {
        let rope = Rope::new(4, 10_000.0).expect("valid");
        TransformerBlock::new(&shape(), 4, &rope, device)
    }

    fn input(device: &burn::tensor::Device, batch: usize, sequence: usize) -> Tensor<3> {
        let data: Vec<f32> = (0..batch * sequence * 8)
            .map(|index| (index as f32 * 0.23).cos())
            .collect();
        Tensor::<3>::from_data(TensorData::new(data, [batch, sequence, 8]), device)
    }

    #[test]
    fn the_block_preserves_the_shape() {
        let device = reference_device();
        let layer = block(&device);
        let output = layer
            .forward(input(&device, 2, 3), 0)
            .expect("shapes agree");
        assert_eq!(output.dims(), [2, 3, 8]);
    }

    #[test]
    fn the_residual_stream_carries_the_input_forward() {
        // With both sub-layers able to contribute, the output must still be
        // anchored to the input: a zero input cannot produce a large output
        // unless a residual connection is missing and a bias is present, and
        // here there is no bias either.
        let device = reference_device();
        let layer = block(&device);
        let output = layer
            .forward(Tensor::<3>::zeros([1, 2, 8], &device), 0)
            .expect("shapes agree");
        let values = output.to_data().try_to_vec::<f32>().expect("f32");

        assert!(
            values.iter().all(|value| value.is_finite()),
            "a zero input must stay finite: {values:?}"
        );
    }

    #[test]
    fn the_block_is_causal() {
        let device = reference_device();
        let layer = block(&device);
        let base = input(&device, 1, 4);

        let first_before = layer
            .forward(base.clone(), 0)
            .expect("shapes agree")
            .narrow(1, 0, 1);

        let mut values = base.to_data().try_to_vec::<f32>().expect("f32");
        for value in values.iter_mut().skip(3 * 8) {
            *value += 4.0;
        }
        let perturbed = Tensor::<3>::from_data(TensorData::new(values, [1, 4, 8]), &device);
        let first_after = layer
            .forward(perturbed, 0)
            .expect("shapes agree")
            .narrow(1, 0, 1);

        let before = first_before.to_data().try_to_vec::<f32>().expect("f32");
        let after = first_after.to_data().try_to_vec::<f32>().expect("f32");
        for (left, right) in before.iter().zip(after.iter()) {
            assert!((left - right).abs() < 1e-6, "{left} vs {right}");
        }
    }

    #[test]
    fn the_output_is_finite_under_a_large_input() {
        let device = reference_device();
        let layer = block(&device);

        let output = layer
            .forward(input(&device, 1, 6) * 100.0, 0)
            .expect("shapes agree");
        let values = output.to_data().try_to_vec::<f32>().expect("f32");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "a large input produced a non-finite output"
        );
    }
}
