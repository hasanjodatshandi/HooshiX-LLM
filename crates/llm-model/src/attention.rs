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
//!
//! # Two ways to score, one answer
//!
//! [`AttentionMode::Whole`] materialises the full
//! `[batch, heads, sequence, key_length]` score block in one tensor. That block
//! is 1.6 GB per layer at the configured Stage-1 shape.
//!
//! [`AttentionMode::Burn`] hands the same block to `burn-tensor`'s own attention,
//! which each backend implements — `burn-flex` picks between a whole-block kernel
//! and a KV-tiled flash kernel by the size of the score block, and the CubeCL
//! backends call a fused kernel. The two modes are the same function; the tests
//! below compare their logits and their gradients element by element.
//!
//! [`AttentionMode::Whole`] stays the default. `Burn` measures **1% slower on
//! `flex` at 2 x 256** and equal at the shapes where both completed, which on a
//! machine that drifted 53% during the session that measured it is noise. The
//! default decision needs a re-measurement on a quiet machine; if it then
//! measures equal or faster, switching is right, because Burn's attention is
//! less code here and is covered by Burn's own backend test suite rather than by
//! this crate's.
//!
//! # The tiled path that was removed
//!
//! A hand-written block-tiled attention lived here and was **correct and
//! measured**: it reproduced this path's logits and gradients. It was removed
//! because it lost on every comparable shape — 6%, 42% and 80% slower on `flex`
//! at 2 x 256, 1 x 512 and 8 x 512 — and because it did **not** raise the GPU
//! memory ceiling, which is what it existed for: every 1024-token shape runs out
//! of memory on its first step on every path, including one that never
//! materialises the score matrix. `docs/11_BENCHMARKING.md` section 17 records
//! the measurement and the four faults found on the way, two of which are traps
//! in Burn itself.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor};

use crate::causal_mask::causal_mask;
use crate::error::ModelError;
use crate::init::weight_initializer;
use crate::kv_cache::LayerCache;
use crate::rope::Rope;

/// Which way attention scores the sequence.
///
/// A closed set, so it is an enum rather than a trait: there are two ways to do
/// this and no third party is expected to supply one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionMode {
    /// Score every query against every key at once, in this crate's own code.
    ///
    /// The whole `[batch, heads, sequence, key_length]` block exists for the
    /// duration of the step. This is the reference and the default.
    #[default]
    Whole,
    /// Hand the block to Burn's own attention, which every backend implements.
    ///
    /// `burn-flex` selects between a whole-block kernel and a KV-tiled flash
    /// kernel by the size of the score block, and the CubeCL backends (`cpu`,
    /// `wgpu`, `cuda`) call a fused kernel. This is the only one of the two whose
    /// arithmetic does not live in this repository, and it is the one the
    /// project's own rule about not building a parallel tensor framework points
    /// at.
    Burn,
}

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
    /// Which way this layer scores. Not a parameter and not persisted, so it is
    /// skipped by the module traversal and by the checkpoint record.
    #[module(skip)]
    mode: AttentionMode,
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
            mode: AttentionMode::default(),
        }
    }

    /// Returns this layer scoring the sequence `mode`'s way.
    ///
    /// The default is [`AttentionMode::Whole`], so nothing that does not ask for
    /// Burn's attention gets it. See the module documentation for why the default
    /// has not moved.
    #[must_use]
    pub fn with_mode(mut self, mode: AttentionMode) -> Self {
        self.mode = mode;
        self
    }

    /// The way this layer scores the sequence.
    #[must_use]
    pub const fn mode(&self) -> AttentionMode {
        self.mode
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
        let attended = match self.mode {
            AttentionMode::Whole => self.attend_whole(query, key, value, position_offset),
            AttentionMode::Burn => attend_burn(query, key, value, position_offset),
        };
        self.merge_heads(attended)
    }

    /// Scores the whole block at once, so the full score tensor exists.
    fn attend_whole(
        &self,
        query: Tensor<4>,
        key: Tensor<4>,
        value: Tensor<4>,
        position_offset: usize,
    ) -> Tensor<4> {
        let [_, _, sequence, _] = query.dims();
        let key_length = key.dims()[2];

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = query.matmul(key.swap_dims(2, 3)) * scale;
        // Built on the device the scores already live on, so no transfer is
        // implied by masking.
        let mask = causal_mask(sequence, key_length, position_offset, &scores.device());
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);

        let weights = burn::tensor::activation::softmax(scores, 3);
        weights.matmul(value)
    }

    /// Rejoins the heads and projects back to the model width.
    fn merge_heads(&self, attended: Tensor<4>) -> Tensor<3> {
        let [batch, _, sequence, _] = attended.dims();
        let merged =
            attended
                .swap_dims(1, 2)
                .reshape([batch, sequence, self.num_heads * self.head_dim]);
        self.output.forward(merged)
    }
}

/// Scores the block with Burn's own attention, which every backend implements.
///
/// `query` is `[batch, heads, sequence, head_dim]`, `key` and `value` are
/// `[batch, heads, key_length, head_dim]` with `key_length >= sequence`, and
/// `position_offset` is the absolute position of the first query against the
/// first key — zero for a full sequence, and `key_length - sequence` for the
/// decode step of a cache. Returns `[batch, heads, sequence, head_dim]` before
/// the output projection.
///
/// # Why this is here
///
/// The project rule is that Burn owns tensor behaviour unless a measured gap
/// requires an extension, and this module hand-rolled attention before anyone
/// checked whether the gap existed. It does not: `burn-tensor` exposes
/// `module::attention`, `burn-flex` chooses between a whole-block kernel and a
/// KV-tiled flash kernel by the size of the score block, and the CubeCL backends
/// call a fused kernel. This function is how the model reaches it.
///
/// It is an **exact-equivalence** result, not a speedup: at every shape tested
/// this reproduces [`attend_whole`](MultiHeadAttention::attend_whole)'s output
/// with a difference of `0.0`, and it measures no faster. See
/// `docs/11_BENCHMARKING.md` section 17.
///
/// # The scale
///
/// Burn defaults the scale to `1 / sqrt(head_dim)` when the options leave it
/// unset, which is the same value this module uses, so no scale is passed and
/// the two cannot drift apart.
///
/// # The mask
///
/// Burn's mask convention is this crate's: `true` marks a position to be set to
/// `-inf` before the softmax, so [`causal_mask`] can be handed over unchanged.
/// The mask is passed rather than Burn's `is_causal` flag because the flag
/// derives its diagonal from `key_length - sequence`, which is only the right
/// answer for a cache that spans the whole history; a mask built from the
/// caller's `position_offset` is right for both. Passing the mask also keeps one
/// definition of causality in the repository instead of two that agree by
/// coincidence.
pub fn attend_burn(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    position_offset: usize,
) -> Tensor<4> {
    let [_, _, sequence, _] = query.dims();
    let key_length = key.dims()[2];
    let mask = causal_mask(sequence, key_length, position_offset, &query.device());

    burn::tensor::module::attention(
        query,
        key,
        value,
        Some(mask),
        None,
        burn::tensor::ops::AttentionModuleOptions::default(),
    )
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

    // --- Burn's attention against this crate's -------------------------------
    //
    // Every test below compares two ways of computing the same function, so the
    // comparison is the assertion.

    use super::AttentionMode;
    use burn::tensor::Device;

    /// Head count and width for the comparisons. Two heads of width four is the
    /// smallest arrangement where a head mix-up cannot pass as agreement.
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 4;

    /// A layer in `mode`, built from the same draw as any other layer.
    fn in_mode(mode: AttentionMode, device: &Device) -> MultiHeadAttention {
        attention(HEADS, HEAD_DIM, device).with_mode(mode)
    }

    /// Deterministic queries, keys and values for one comparison.
    ///
    /// `magnitude` scales the draw. It matters: a layer's projections of an
    /// input of order one produce scores of order `1e-3`, and a softmax over
    /// scores that flat is a uniform average whose weights are decided by the
    /// last bit of `exp`. Comparisons are therefore run at a magnitude that
    /// spreads the scores over a range the softmax actually resolves, as well as
    /// at the natural one.
    fn operands_scaled(
        device: &Device,
        batch: usize,
        heads: usize,
        sequence: usize,
        key_length: usize,
        head_dim: usize,
        magnitude: f32,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let draw = |offset: f32, rows: usize| {
            let data: Vec<f32> = (0..batch * heads * rows * head_dim)
                .map(|index| magnitude * ((index as f32 + offset) * 0.173).sin())
                .collect();
            Tensor::<4>::from_data(
                TensorData::new(data, [batch, heads, rows, head_dim]),
                device,
            )
        };
        (
            draw(0.0, sequence),
            draw(11.0, key_length),
            draw(23.0, key_length),
        )
    }

    /// The same draw at the natural magnitude of one.
    fn operands(
        device: &Device,
        batch: usize,
        heads: usize,
        sequence: usize,
        key_length: usize,
        head_dim: usize,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        operands_scaled(device, batch, heads, sequence, key_length, head_dim, 1.0)
    }

    /// The output of the whole-matrix path for one shape.
    fn whole(
        device: &Device,
        batch: usize,
        sequence: usize,
        key_length: usize,
        position_offset: usize,
    ) -> Vec<f32> {
        let (query, key, value) = operands(device, batch, HEADS, sequence, key_length, HEAD_DIM);
        in_mode(AttentionMode::Whole, device)
            .attend_whole(query, key, value, position_offset)
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32")
    }

    /// Shapes worth comparing, as `(sequence, key_length, position_offset)`.
    ///
    /// Every one satisfies `position_offset + sequence <= key_length`, which is
    /// the cache contract: a decode step attends `sequence` new queries to the
    /// `position_offset + sequence` keys it has already seen.
    const SHAPES: [(usize, usize, usize); 5] =
        [(8, 8, 0), (3, 6, 3), (10, 10, 0), (20, 20, 0), (17, 20, 3)];

    #[test]
    fn burn_attention_produces_what_the_whole_matrix_path_produces() {
        // Burn's attention is a backend-dispatched block, so it is only usable
        // if it computes this model's function. Every shape below is compared,
        // including the decode offsets where the keys outnumber the queries.
        use super::attend_burn;

        let device = reference_device();

        for (sequence, key_length, position_offset) in SHAPES {
            let expected = whole(&device, 1, sequence, key_length, position_offset);
            let (query, key, value) = operands(&device, 1, HEADS, sequence, key_length, HEAD_DIM);
            let produced = attend_burn(query, key, value, position_offset)
                .to_data()
                .try_to_vec::<f32>()
                .expect("f32");

            assert_eq!(
                produced.len(),
                expected.len(),
                "sequence {sequence}, keys {key_length}, offset {position_offset}: \
                 Burn's attention returned a different shape"
            );
            assert!(
                produced.iter().all(|value| value.is_finite()),
                "sequence {sequence}, keys {key_length}, offset {position_offset}: \
                 Burn's attention returned a non-finite value"
            );
            let worst = produced
                .iter()
                .zip(expected.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                worst < 1e-5,
                "sequence {sequence}, keys {key_length}, offset {position_offset}: \
                 Burn's attention differs from the whole path by {worst}"
            );
        }
    }

    #[test]
    fn the_burn_layer_is_what_the_default_layer_is() {
        let device = reference_device();
        for sequence in [1_usize, 3, 8, 20, 65] {
            let projected = || input(&device, 1, sequence, HEADS * HEAD_DIM) * 40.0;
            let reference = in_mode(AttentionMode::Whole, &device);
            let burn_layer = reference.clone().with_mode(AttentionMode::Burn);

            let expected = reference
                .forward(projected(), 0)
                .expect("shapes agree")
                .to_data()
                .try_to_vec::<f32>()
                .expect("f32");
            let produced = burn_layer
                .forward(projected(), 0)
                .expect("shapes agree")
                .to_data()
                .try_to_vec::<f32>()
                .expect("f32");

            assert!(
                produced.iter().all(|value| value.is_finite()),
                "sequence {sequence}: the Burn path produced a non-finite logit"
            );
            let worst = produced
                .iter()
                .zip(expected.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                worst < 1e-4,
                "sequence {sequence}: the Burn path's logits differ by {worst}"
            );
        }
    }

    #[test]
    fn the_burn_path_trains_the_same_as_the_whole_one() {
        // A forward pass that agrees proves nothing about the backward pass, and
        // this is a path whose backward lives outside this crate.
        use super::attend_burn;

        let device = reference_device().autodiff();
        let data: Vec<f32> = (0..(2 * HEADS * 20 * HEAD_DIM))
            .map(|index| 40.0 * (index as f32 * 0.0173).sin())
            .collect();
        let shape = [2, HEADS, 20, HEAD_DIM];

        // One leaf serving as query, key and value at once, so the single
        // gradient that comes back is the sum of the three and every operand has
        // to have been differentiated for it to match.
        let leaf = Tensor::<4>::from_data(TensorData::new(data, shape), &device).require_grad();

        let layer = in_mode(AttentionMode::Whole, &device);
        let whole_gradients = layer
            .attend_whole(leaf.clone(), leaf.clone(), leaf.clone(), 0)
            .sum()
            .backward();
        let burn_gradients = attend_burn(leaf.clone(), leaf.clone(), leaf.clone(), 0)
            .sum()
            .backward();

        let expected = leaf
            .grad(&whole_gradients)
            .expect("the whole path must reach the leaf")
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");
        let produced = leaf
            .grad(&burn_gradients)
            .expect("the Burn path must reach the leaf")
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");

        assert!(
            produced.iter().all(|value| value.is_finite()),
            "the Burn path's gradient is not finite"
        );
        assert!(
            produced.iter().any(|value| value.abs() > 1e-9),
            "the Burn path's gradient is entirely zero, so nothing reached it"
        );
        assert!(
            expected.iter().any(|value| value.abs() > 1e-9),
            "the whole path's gradient is entirely zero, so the comparison is empty"
        );
        // A relative bound, because the two backward passes sum the same terms
        // in different orders: Burn's goes through backend primitives and this
        // crate's through the autodiff graph, so the last bits differ. The
        // forward comparison above is the one that can be absolute, and it is.
        let largest = expected
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(value.abs()));
        let worst = produced
            .iter()
            .zip(expected.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst <= 1e-5 + 3e-5 * largest,
            "the gradient differs by {worst}, against a largest expected value of {largest}"
        );
    }

    /// An installer that gives any layer `donor`'s drawn weights.
    ///
    /// The draw is deferred to first read and the reference backend shares one
    /// random source across the process, so `init::seed` does not make two
    /// separately built layers equal: another test can read a parameter in
    /// between and advance the source. Copying the drawn values removes the
    /// question, so the comparison measures the two scoring paths and nothing
    /// else. Reading `val()` here rather than later is what forces the draw.
    fn weight_installer<'a>(
        donor: &MultiHeadAttention,
        device: &'a Device,
    ) -> impl FnMut(MultiHeadAttention) -> MultiHeadAttention + 'a {
        let values: Vec<TensorData> = [
            &donor.query.weight,
            &donor.key.weight,
            &donor.value.weight,
            &donor.output.weight,
        ]
        .iter()
        .map(|weight| weight.val().to_data())
        .collect();

        move |mut layer: MultiHeadAttention| {
            // `require_grad` is not optional here: a tensor rebuilt with
            // `from_data` on an autodiff device is not a tracked leaf, and
            // without it the parameter silently drops out of the graph.
            let install = |weight: &mut burn::module::Param<Tensor<2>>, index: usize| {
                let taken = weight.clone();
                *weight =
                    taken.map(|_| Tensor::from_data(values[index].clone(), device).require_grad());
            };
            install(&mut layer.query.weight, 0);
            install(&mut layer.key.weight, 1);
            install(&mut layer.value.weight, 2);
            install(&mut layer.output.weight, 3);
            layer
        }
    }

    /// Gradient of the sum of attention outputs, for every projection `Linear`.
    fn gradients_in_mode(
        device: &Device,
        mode: AttentionMode,
        batch: usize,
        sequence: usize,
        install: &mut dyn FnMut(MultiHeadAttention) -> MultiHeadAttention,
    ) -> Vec<Vec<f32>> {
        let layer = install(in_mode(mode, device));
        let objective = layer
            .forward(input(device, batch, sequence, HEADS * HEAD_DIM), 0)
            .expect("shapes agree")
            .sum();
        let gradients = objective.backward();

        [&layer.query, &layer.key, &layer.value, &layer.output]
            .iter()
            .map(|projection| {
                let gradient = projection
                    .weight
                    .grad(&gradients)
                    .unwrap_or_else(|| panic!("a projection weight must be in the graph"));
                gradient.to_data().try_to_vec::<f32>().expect("f32")
            })
            .collect()
    }

    #[test]
    fn the_burn_path_trains_every_projection_the_same_as_the_whole_one() {
        // Logits that agree say nothing about the backward pass, and a training
        // run's correctness lives there. The same batch goes through the same
        // objective down both paths, on the same weights, and every projection's
        // gradient is compared element by element.
        let device = reference_device().autodiff();
        let names = ["query", "key", "value", "output"];

        for (batch, sequence) in [(1_usize, 20_usize), (1, 28), (2, 10), (2, 17)] {
            // One draw, installed into both layers, so the two differ in the
            // scoring path and in nothing else.
            let donor = in_mode(AttentionMode::Whole, &device);
            let mut install = weight_installer(&donor, &device);

            let whole_gradients =
                gradients_in_mode(&device, AttentionMode::Whole, batch, sequence, &mut install);
            let burn_gradients =
                gradients_in_mode(&device, AttentionMode::Burn, batch, sequence, &mut install);

            for (index, name) in names.iter().enumerate() {
                let expected = &whole_gradients[index];
                let produced = &burn_gradients[index];
                assert_eq!(
                    produced.len(),
                    expected.len(),
                    "batch {batch}, sequence {sequence}: the {name} gradient changed size"
                );
                assert!(
                    produced.iter().all(|value| value.is_finite()),
                    "batch {batch}, sequence {sequence}: the Burn {name} gradient is not finite"
                );
                assert!(
                    expected.iter().any(|value| value.abs() > 1e-12),
                    "batch {batch}, sequence {sequence}: the whole {name} gradient is entirely \
                     zero, so the comparison is empty"
                );
                // A relative bound: the two backward passes sum the same terms in
                // different orders, so the last bits differ. See the other
                // gradient comparison for the same reasoning.
                let largest = expected
                    .iter()
                    .fold(0.0_f32, |acc, value| acc.max(value.abs()));
                let worst = produced
                    .iter()
                    .zip(expected.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    worst <= 1e-5 + 3e-5 * largest,
                    "batch {batch}, sequence {sequence}: the {name} gradient differs by {worst}, \
                     against a largest expected value of {largest}"
                );
            }
        }
    }
}
