//! Rotary position embedding.
//!
//! Rotary embedding rotates each adjacent channel pair of the query and key
//! vectors by an angle proportional to the token's position, so attention
//! scores depend on relative distance without any added parameters.
//!
//! Channel pairs are `(2i, 2i+1)`. This is the original formulation from the
//! rotary-embedding paper, and it is stated here because the alternative
//! half-split convention is not interchangeable: a model trained under one and
//! evaluated under the other produces meaningless scores.
//!
//! The angles are computed with tensor operations on the device rather than
//! from a host-side table, so a training step does not transfer a table per
//! call. A cached table remains a possible optimization, but it would need its
//! own equivalence test and bounded-memory argument.

use burn::module::Module;
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::error::ModelError;

/// Applies rotary embedding to queries and keys of shape
/// `[batch, heads, sequence, head_dim]`.
///
/// The embedding holds no parameters, but it is a `Module` so that an attention
/// layer can own one without a special case for parameter-free components.
#[derive(Module, Debug)]
pub struct Rope {
    head_dim: usize,
    theta: f64,
}

impl Rope {
    /// Creates the embedding.
    ///
    /// The rotary dimension must be even, because the rotation operates on
    /// channel pairs.
    pub fn new(head_dim: usize, theta: f64) -> Result<Self, ModelError> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(ModelError::InvalidConfig {
                reason: format!("rotary head dimension must be even and non-zero, got {head_dim}"),
            });
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(ModelError::InvalidConfig {
                reason: format!("rotary base period must be finite and positive, got {theta}"),
            });
        }
        Ok(Self { head_dim, theta })
    }

    /// Rotary dimension this embedding was built for.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Base period.
    #[must_use]
    pub const fn theta(&self) -> f64 {
        self.theta
    }

    /// Rotates a query and a key that share one shape.
    ///
    /// This is the self-attention form: one call for a pair of
    /// `[batch, heads, sequence, head_dim]` tensors.
    ///
    /// `position_offset` is the absolute position of the first token in the
    /// sequence. Cached decoding passes the length of the cache so that a
    /// one-token step is rotated as if it sat at its true position, which is
    /// what makes cached and uncached decoding agree. When the two sides have
    /// different lengths, rotate each with [`Rope::rotate`] instead.
    pub fn forward(
        &self,
        query: Tensor<4>,
        key: Tensor<4>,
        position_offset: usize,
    ) -> Result<(Tensor<4>, Tensor<4>), ModelError> {
        let query_dims = query.dims();
        let key_dims = key.dims();

        if query_dims != key_dims {
            return Err(ModelError::Shape {
                operation: "rotary position embedding of a query and key pair",
                expected: format!("query and key of the same shape, query is {query_dims:?}"),
                actual: format!("key is {key_dims:?}"),
            });
        }

        Ok((
            self.rotate(query, position_offset)?,
            self.rotate(key, position_offset)?,
        ))
    }

    /// Rotates one tensor of shape `[batch, heads, sequence, head_dim]`.
    ///
    /// This is the primitive. It is separate from [`Rope::forward`] because a
    /// query and a key set do not always have the same length: scoring a single
    /// query against many keys rotates each side once, at its own length.
    pub fn rotate(
        &self,
        tensor: Tensor<4>,
        position_offset: usize,
    ) -> Result<Tensor<4>, ModelError> {
        let [_, _, sequence, head_dim] = tensor.dims();

        if head_dim != self.head_dim {
            return Err(ModelError::Shape {
                operation: "rotary position embedding",
                expected: format!("head dimension {}", self.head_dim),
                actual: format!("head dimension {head_dim}"),
            });
        }

        let device = tensor.device();
        let (cosine, sine) = self.angle_tables(sequence, position_offset, &device);
        Ok(self.apply(tensor, cosine, sine))
    }

    /// Builds the cosine and sine tables for `sequence` positions.
    ///
    /// Both are shaped `[1, 1, sequence, head_dim / 2]`, ready to broadcast
    /// against a `[batch, heads, sequence, head_dim / 2]` slice.
    fn angle_tables(
        &self,
        sequence: usize,
        position_offset: usize,
        device: &Device,
    ) -> (Tensor<4>, Tensor<4>) {
        let half = self.head_dim / 2;

        // The inverse frequencies are a function of the head dimension alone, so
        // computing them on the host is exact and costs one small transfer per
        // call rather than one per step of the sequence.
        let inverse_frequencies: Vec<f32> = (0..half)
            .map(|index| {
                let exponent = 2.0 * index as f64 / self.head_dim as f64;
                (1.0 / self.theta.powf(exponent)) as f32
            })
            .collect();
        let inverse_frequencies =
            Tensor::<1>::from_data(TensorData::new(inverse_frequencies, [half]), device);

        let start = i64::try_from(position_offset).unwrap_or(i64::MAX);
        let end = start.saturating_add(sequence as i64);
        let positions = Tensor::<1, Int>::arange(start..end, device).float();

        // [sequence, 1] * [1, half] broadcasts to [sequence, half].
        let angles = positions.unsqueeze_dim::<2>(1) * inverse_frequencies.unsqueeze_dim::<2>(0);

        let cosine = angles.clone().cos().unsqueeze::<3>().unsqueeze::<4>();
        let sine = angles.sin().unsqueeze::<3>().unsqueeze::<4>();
        (cosine, sine)
    }

    /// Rotates adjacent channel pairs of one tensor.
    fn apply(&self, tensor: Tensor<4>, cosine: Tensor<4>, sine: Tensor<4>) -> Tensor<4> {
        let [batch, heads, sequence, head_dim] = tensor.dims();
        let half = head_dim / 2;

        // Splitting the last axis into pairs makes the even and odd channels
        // adjacent, which is what "adjacent pair" means here.
        let pairs = tensor.reshape([batch, heads, sequence, half, 2]);
        let even = pairs
            .clone()
            .narrow(4, 0, 1)
            .reshape([batch, heads, sequence, half]);
        let odd = pairs
            .narrow(4, 1, 1)
            .reshape([batch, heads, sequence, half]);

        let rotated_even = even.clone() * cosine.clone() - odd.clone() * sine.clone();
        let rotated_odd = even * sine + odd * cosine;

        Tensor::cat(
            vec![
                rotated_even.unsqueeze_dim::<5>(4),
                rotated_odd.unsqueeze_dim::<5>(4),
            ],
            4,
        )
        .reshape([batch, heads, sequence, head_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::Rope;
    use crate::error::ModelError;
    use crate::testing::reference_device;
    use burn::tensor::{Tensor, TensorData};

    fn tensor(values: Vec<f32>, shape: [usize; 4], device: &burn::tensor::Device) -> Tensor<4> {
        Tensor::<4>::from_data(TensorData::new(values, shape), device)
    }

    fn values(tensor: &Tensor<4>) -> Vec<f32> {
        tensor
            .to_data()
            .try_to_vec::<f32>()
            .expect("the tensor holds f32")
    }

    /// Host-side reference for one rotation, following the documented formula.
    fn reference_rotate(
        input: &[f32],
        sequence: usize,
        head_dim: usize,
        theta: f64,
        offset: usize,
    ) -> Vec<f32> {
        let half = head_dim / 2;
        let mut out = input.to_vec();
        for position in 0..sequence {
            for index in 0..half {
                let exponent = 2.0 * index as f64 / head_dim as f64;
                let angle = (offset + position) as f64 * theta.powf(-exponent);
                let (sin, cos) = angle.sin_cos();
                let even_index = position * head_dim + 2 * index;
                let odd_index = even_index + 1;
                let even = f64::from(input[even_index]);
                let odd = f64::from(input[odd_index]);
                out[even_index] = (even * cos - odd * sin) as f32;
                out[odd_index] = (even * sin + odd * cos) as f32;
            }
        }
        out
    }

    #[test]
    fn an_odd_rotary_dimension_is_rejected() {
        assert!(matches!(
            Rope::new(5, 10_000.0),
            Err(ModelError::InvalidConfig { .. })
        ));
        assert!(matches!(
            Rope::new(0, 10_000.0),
            Err(ModelError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn a_non_positive_base_period_is_rejected() {
        for theta in [0.0, -1.0, f64::NAN] {
            assert!(matches!(
                Rope::new(8, theta),
                Err(ModelError::InvalidConfig { .. })
            ));
        }
    }

    #[test]
    fn position_zero_leaves_the_first_token_unrotated() {
        let device = reference_device();
        let rope = Rope::new(4, 10_000.0).expect("valid");
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let query = tensor(input.clone(), [1, 1, 1, 4], &device);
        let key = tensor(input.clone(), [1, 1, 1, 4], &device);
        let (rotated, _) = rope.forward(query, key, 0).expect("shapes agree");

        // At position zero every angle is zero, so the rotation is the identity.
        for (left, right) in values(&rotated).iter().zip(input.iter()) {
            assert!((left - right).abs() < 1e-6, "{left} vs {right}");
        }
    }

    #[test]
    fn matches_a_host_side_reference_implementation() {
        let device = reference_device();
        let head_dim = 8;
        let sequence = 5;
        let rope = Rope::new(head_dim, 10_000.0).expect("valid");

        let input: Vec<f32> = (0..sequence * head_dim)
            .map(|index| (index as f32 * 0.29).cos())
            .collect();
        let query = tensor(input.clone(), [1, 1, sequence, head_dim], &device);
        let key = tensor(input.clone(), [1, 1, sequence, head_dim], &device);

        let (rotated, _) = rope.forward(query, key, 0).expect("shapes agree");
        let expected = reference_rotate(&input, sequence, head_dim, 10_000.0, 0);

        for (index, (left, right)) in values(&rotated).iter().zip(expected.iter()).enumerate() {
            assert!(
                (left - right).abs() < 1e-5,
                "element {index}: {left} vs {right}"
            );
        }
    }

    #[test]
    fn the_position_offset_shifts_the_rotation() {
        let device = reference_device();
        let head_dim = 4;
        let rope = Rope::new(head_dim, 10_000.0).expect("valid");
        let input: Vec<f32> = (0..head_dim).map(|index| index as f32 + 1.0).collect();

        let query = tensor(input.clone(), [1, 1, 1, head_dim], &device);
        let key = tensor(input.clone(), [1, 1, 1, head_dim], &device);
        let (rotated, _) = rope.forward(query, key, 7).expect("shapes agree");

        let expected = reference_rotate(&input, 1, head_dim, 10_000.0, 7);
        for (left, right) in values(&rotated).iter().zip(expected.iter()) {
            assert!((left - right).abs() < 1e-6, "{left} vs {right}");
        }
    }

    #[test]
    fn a_rotation_preserves_each_pair_norm() {
        let device = reference_device();
        let head_dim = 6;
        let sequence = 4;
        let rope = Rope::new(head_dim, 10_000.0).expect("valid");

        let input: Vec<f32> = (0..sequence * head_dim)
            .map(|index| (index as f32 * 0.71).sin())
            .collect();
        let query = tensor(input.clone(), [1, 1, sequence, head_dim], &device);
        let key = tensor(input.clone(), [1, 1, sequence, head_dim], &device);
        let (rotated, _) = rope.forward(query, key, 0).expect("shapes agree");
        let actual = values(&rotated);

        for position in 0..sequence {
            for pair in 0..head_dim / 2 {
                let base = position * head_dim + 2 * pair;
                let before = input[base].hypot(input[base + 1]);
                let after = actual[base].hypot(actual[base + 1]);
                assert!(
                    (before - after).abs() < 1e-5,
                    "pair norm changed at position {position}, pair {pair}: {before} vs {after}"
                );
            }
        }
    }

    #[test]
    fn a_head_dimension_mismatch_is_reported() {
        let device = reference_device();
        let rope = Rope::new(8, 10_000.0).expect("valid");

        let query = tensor(vec![0.0; 4], [1, 1, 1, 4], &device);
        let key = tensor(vec![0.0; 4], [1, 1, 1, 4], &device);

        assert!(matches!(
            rope.forward(query, key, 0),
            Err(ModelError::Shape { .. })
        ));
    }

    #[test]
    fn unequal_query_and_key_shapes_are_reported() {
        let device = reference_device();
        let rope = Rope::new(4, 10_000.0).expect("valid");

        let query = tensor(vec![0.0; 8], [1, 1, 2, 4], &device);
        let key = tensor(vec![0.0; 4], [1, 1, 1, 4], &device);

        assert!(matches!(
            rope.forward(query, key, 0),
            Err(ModelError::Shape { .. })
        ));
    }

    #[test]
    fn attention_scores_depend_only_on_relative_distance() {
        // With head_dim 2 there is a single frequency, so the score between a
        // query at one position and a key at another is determined by the
        // distance between them, not by where either sits. A query and a key at
        // the same position must also keep a score of one, because a rotation
        // applied to both leaves their inner product alone.
        let device = reference_device();
        let head_dim = 2;
        let sequence = 4;
        let rope = Rope::new(head_dim, 10_000.0).expect("valid");

        let query = tensor(vec![1.0, 0.0], [1, 1, 1, head_dim], &device);
        let key = tensor(
            vec![
                1.0, 0.0, // position 0
                1.0, 0.0, // position 1
                1.0, 0.0, // position 2
                1.0, 0.0, // position 3
            ],
            [1, 1, sequence, head_dim],
            &device,
        );

        let scores_at = |offset: usize| -> Vec<f32> {
            // The query covers one position and the keys cover four, which is
            // why rotation is applied per tensor rather than through the
            // equal-shape convenience form.
            let rotated_query = rope.rotate(query.clone(), offset).expect("shapes agree");
            let rotated_key = rope.rotate(key.clone(), offset).expect("shapes agree");

            // [1, 1, 1, 2] * [1, 1, 4, 2] broadcasts to one score per key.
            (rotated_query * rotated_key)
                .sum_dim(3)
                .reshape([sequence])
                .to_data()
                .try_to_vec::<f32>()
                .expect("the tensor holds f32")
        };

        let from_zero = scores_at(0);
        let from_five = scores_at(5);

        // Same distances, shifted origin: the scores must be identical.
        for (index, (left, right)) in from_zero.iter().zip(from_five.iter()).enumerate() {
            assert!(
                (left - right).abs() < 1e-5,
                "distance {index}: {left} at offset 0 versus {right} at offset 5"
            );
        }

        // The query against the key at its own position is untouched.
        assert!((from_zero[0] - 1.0).abs() < 1e-6);
        // With a single frequency of one radian, the score falls as the
        // distance grows.
        assert!(
            from_zero[0] > from_zero[1] && from_zero[1] > from_zero[2],
            "scores must fall with distance: {from_zero:?}"
        );
    }
}
