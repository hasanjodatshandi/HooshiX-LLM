//! Multi-head causal self-attention.
//!
//! Conceptual form, from the model specification:
//!
//! ```text
//! Q = X Wq     K = X Wk     V = X Wv
//! scores = (Q K^T) / sqrt(head_dim) + causal_mask
//! P = softmax(scores)
//! A = P V
//! Y = concat(A_heads) Wo
//! ```
//!
//! Stage 1 is plain multi-head attention: every head has its own key and value
//! projection. Grouped-query attention is a Stage-2 change and is refused here
//! rather than approximated, so the architecture cannot drift by accident.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor};

use crate::causal_mask::causal_mask;
use crate::error::ModelError;
use crate::init::weight_initializer;
use crate::kv_cache::LayerCache;
use crate::rope::Rope;

/// Self-attention over a sequence, with rotary positions and a causal mask.
#[derive(Module, Debug)]
pub struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    rope: Rope,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    /// Creates the layer.
    #[must_use]
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        bias: bool,
        rope: &Rope,
        device: &Device,
    ) -> Self {
        let project = |from: usize, to: usize| {
            LinearConfig::new(from, to)
                .with_bias(bias)
                .with_initializer(weight_initializer())
                .init(device)
        };
        let width = num_heads * head_dim;
        Self {
            query: project(hidden_size, width),
            key: project(hidden_size, width),
            value: project(hidden_size, width),
            output: project(width, hidden_size),
            rope: rope.clone(),
            num_heads,
            head_dim,
        }
    }

    /// Number of attention heads.
    #[must_use]
    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Width of one head.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Width of the concatenated projections, `num_heads * head_dim`.
    ///
    /// A Burn `Linear` weight is `[input, output]`, so the output width is the
    /// second axis. Exposed so a test can check the projection width against a
    /// hidden size the two do not share.
    #[must_use]
    pub fn projection_width(&self) -> usize {
        self.query.weight.val().dims()[1]
    }

    /// Applies attention to `[batch, sequence, hidden]`.
    ///
    /// `position_offset` is the absolute position of the first token, so a
    /// cached single-token step is rotated as if it sat at its true place.
    pub fn forward(
        &self,
        input: Tensor<3>,
        position_offset: usize,
    ) -> Result<Tensor<3>, ModelError> {
        let [batch, sequence, hidden] = input.dims();
        self.check_hidden(hidden)?;

        let (query, key, value) = self.project_and_split(input, batch, sequence);
        let (query, key) = self.rope.forward(query, key, position_offset)?;
        Ok(self.attend(query, key, value, position_offset))
    }

    /// Applies attention to `[batch, sequence, hidden]`, extending `cache`.
    ///
    /// This is the incremental-decoding path. `position_offset` must be the
    /// absolute position of the first new token, which is also the number of
    /// positions the cache already holds; anything else would rotate the new
    /// tokens onto the wrong positions or let a query see a key it should not.
    pub fn forward_cached(
        &self,
        input: Tensor<3>,
        cache: &mut LayerCache,
        position_offset: usize,
    ) -> Result<Tensor<3>, ModelError> {
        let [batch, sequence, hidden] = input.dims();
        self.check_hidden(hidden)?;

        let (query, key, value) = self.project_and_split(input, batch, sequence);
        let (query, key) = self.rope.forward(query, key, position_offset)?;

        // The cache returns the whole history, which is what the new queries
        // must attend to.
        let (key, value) = cache.append(key, value)?;
        Ok(self.attend(query, key, value, position_offset))
    }

    /// Rejects an input whose width is not the one this layer projects.
    fn check_hidden(&self, hidden: usize) -> Result<(), ModelError> {
        let expected = self.output.weight.val().dims()[1];
        if hidden == expected {
            Ok(())
        } else {
            Err(ModelError::Shape {
                operation: "multi-head self-attention",
                expected: format!("hidden size {expected}"),
                actual: format!("hidden size {hidden}"),
            })
        }
    }

    /// Projects the input and splits the model axis into heads.
    ///
    /// `[batch, sequence, heads * head_dim]` becomes
    /// `[batch, heads, sequence, head_dim]`.
    fn project_and_split(
        &self,
        input: Tensor<3>,
        batch: usize,
        sequence: usize,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let split = |projected: Tensor<3>| {
            projected
                .reshape([batch, sequence, self.num_heads, self.head_dim])
                .swap_dims(1, 2)
        };

        (
            split(self.query.forward(input.clone())),
            split(self.key.forward(input.clone())),
            split(self.value.forward(input)),
        )
    }

    /// Scores `sequence` queries against however many keys it is handed.
    ///
    /// A cached step has more keys than queries, which is why the mask takes the
    /// two lengths separately rather than assuming a square block.
    fn attend(
        &self,
        query: Tensor<4>,
        key: Tensor<4>,
        value: Tensor<4>,
        position_offset: usize,
    ) -> Tensor<3> {
        let [batch, _, sequence, _] = query.dims();
        let key_length = key.dims()[2];

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = query.matmul(key.swap_dims(2, 3)) * scale;
        // Built on the device the scores already live on, so no transfer is
        // implied by masking.
        let mask = causal_mask(sequence, key_length, position_offset, &scores.device());
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);

        let weights = burn::tensor::activation::softmax(scores, 3);
        let attended = weights.matmul(value);

        // Rejoin the heads and project back to the model width.
        let merged =
            attended
                .swap_dims(1, 2)
                .reshape([batch, sequence, self.num_heads * self.head_dim]);
        self.output.forward(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::MultiHeadAttention;
    use crate::rope::Rope;
    use crate::testing::reference_device;
    use burn::tensor::{Tensor, TensorData};

    fn attention(
        heads: usize,
        head_dim: usize,
        device: &burn::tensor::Device,
    ) -> MultiHeadAttention {
        let rope = Rope::new(head_dim, 10_000.0).expect("valid");
        MultiHeadAttention::new(heads * head_dim, heads, head_dim, false, &rope, device)
    }

    fn input(
        device: &burn::tensor::Device,
        batch: usize,
        sequence: usize,
        hidden: usize,
    ) -> Tensor<3> {
        let data: Vec<f32> = (0..batch * sequence * hidden)
            .map(|index| (index as f32 * 0.37).sin())
            .collect();
        Tensor::<3>::from_data(TensorData::new(data, [batch, sequence, hidden]), device)
    }

    #[test]
    fn the_output_shape_matches_the_input_shape() {
        let device = reference_device();
        let layer = attention(2, 4, &device);

        let output = layer
            .forward(input(&device, 2, 5, 8), 0)
            .expect("shapes agree");
        assert_eq!(output.dims(), [2, 5, 8]);
        assert_eq!(layer.num_heads(), 2);
        assert_eq!(layer.head_dim(), 4);
        assert_eq!(layer.projection_width(), 8);
    }

    #[test]
    fn the_projection_width_is_the_head_count_times_the_head_dimension() {
        // A hidden size that differs from the projection width, so reading the
        // wrong axis of the weight cannot pass by coincidence.
        let device = reference_device();
        let rope = Rope::new(4, 10_000.0).expect("valid");
        let layer = MultiHeadAttention::new(12, 2, 4, false, &rope, &device);

        assert_eq!(layer.projection_width(), 8);

        let output = layer
            .forward(input(&device, 1, 3, 12), 0)
            .expect("shapes agree");
        assert_eq!(output.dims(), [1, 3, 12]);
    }

    #[test]
    fn a_single_token_sequence_is_accepted() {
        let device = reference_device();
        let layer = attention(2, 4, &device);

        let output = layer
            .forward(input(&device, 1, 1, 8), 0)
            .expect("shapes agree");
        assert_eq!(output.dims(), [1, 1, 8]);
    }

    #[test]
    fn the_first_position_cannot_depend_on_later_positions() {
        // Change the last token and watch the first output row. If the causal
        // mask were missing or wrong, the first row would move.
        let device = reference_device();
        let layer = attention(2, 4, &device);

        let base = input(&device, 1, 4, 8);
        let first_before = layer
            .forward(base.clone(), 0)
            .expect("shapes agree")
            .narrow(1, 0, 1);

        let changed = base.to_data();
        let mut values = changed.try_to_vec::<f32>().expect("f32");
        // Overwrite the final token's features.
        for value in values.iter_mut().skip(3 * 8) {
            *value += 5.0;
        }
        let perturbed = Tensor::<3>::from_data(TensorData::new(values, [1, 4, 8]), &device);

        let first_after = layer
            .forward(perturbed, 0)
            .expect("shapes agree")
            .narrow(1, 0, 1);

        let before = first_before.to_data().try_to_vec::<f32>().expect("f32");
        let after = first_after.to_data().try_to_vec::<f32>().expect("f32");
        for (left, right) in before.iter().zip(after.iter()) {
            assert!(
                (left - right).abs() < 1e-6,
                "the first position changed when a later token changed: {left} vs {right}"
            );
        }
    }

    #[test]
    fn a_later_position_does_depend_on_earlier_positions() {
        // The complement of the previous test: the last row must react.
        let device = reference_device();
        let layer = attention(2, 4, &device);

        let base = input(&device, 1, 4, 8);
        let last_before = layer
            .forward(base.clone(), 0)
            .expect("shapes agree")
            .narrow(1, 3, 1);

        let mut values = base.to_data().try_to_vec::<f32>().expect("f32");
        for value in values.iter_mut().take(8) {
            *value += 5.0;
        }
        let perturbed = Tensor::<3>::from_data(TensorData::new(values, [1, 4, 8]), &device);
        let last_after = layer
            .forward(perturbed, 0)
            .expect("shapes agree")
            .narrow(1, 3, 1);

        let before = last_before.to_data().try_to_vec::<f32>().expect("f32");
        let after = last_after.to_data().try_to_vec::<f32>().expect("f32");
        let moved = before
            .iter()
            .zip(after.iter())
            .any(|(left, right)| (left - right).abs() > 1e-6);
        assert!(moved, "a later position must see earlier ones");
    }

    #[test]
    fn changing_the_sequence_length_does_not_move_earlier_outputs() {
        // Causal attention with rotary positions must give the same prefix
        // whether the sequence stops there or continues.
        let device = reference_device();
        let layer = attention(2, 4, &device);
        let full = input(&device, 1, 6, 8);
        let short = full.clone().narrow(1, 0, 3);

        let from_short = layer
            .forward(short, 0)
            .expect("shapes agree")
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");
        let from_full = layer
            .forward(full, 0)
            .expect("shapes agree")
            .narrow(1, 0, 3)
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");

        for (index, (left, right)) in from_short.iter().zip(from_full.iter()).enumerate() {
            assert!(
                (left - right).abs() < 1e-5,
                "element {index}: {left} from the short sequence vs {right} from the long one"
            );
        }
    }

    #[test]
    fn a_hidden_size_mismatch_is_reported() {
        let device = reference_device();
        let layer = attention(2, 4, &device);

        assert!(layer.forward(input(&device, 1, 2, 12), 0).is_err());
    }

    #[test]
    fn the_output_stays_finite_for_large_inputs() {
        let device = reference_device();
        let layer = attention(2, 4, &device);

        for scale in [1.0_f32, 50.0, -50.0] {
            let output = layer
                .forward(input(&device, 1, 8, 8) * scale, 0)
                .expect("shapes agree");
            let values = output.to_data().try_to_vec::<f32>().expect("f32");
            assert!(
                values.iter().all(|value| value.is_finite()),
                "scale {scale} produced a non-finite output"
            );
        }
    }

    #[test]
    fn a_backward_pass_reaches_all_four_projections() {
        let device = reference_device().autodiff();
        let layer = attention(2, 4, &device);

        let objective = layer
            .forward(input(&device, 1, 3, 8), 0)
            .expect("shapes agree")
            .sum();
        let gradients = objective.backward();

        for (name, parameter) in [
            ("query", &layer.query.weight),
            ("key", &layer.key.weight),
            ("value", &layer.value.weight),
            ("output", &layer.output.weight),
        ] {
            let gradient = parameter
                .grad(&gradients)
                .unwrap_or_else(|| panic!("the {name} projection must be in the graph"));
            crate::grad::ensure_finite(&gradient, crate::error::NumericalQuantity::Gradients)
                .unwrap_or_else(|error| panic!("the {name} gradient is not finite: {error}"));
        }
    }
}
