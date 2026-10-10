//! Root-mean-square layer normalization.
//!
//! Reference form, from the model specification:
//!
//! ```text
//! rms(x) = sqrt(mean(x^2) + eps)
//! y      = x / rms(x) * weight
//! ```
//!
//! The normalization is computed over the last axis, in the dtype of the input.
//! It is written with ordinary tensor operations rather than a fused kernel so
//! that the reference path stays readable and testable.

use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::tensor::{Device, Tensor};

/// A normalization layer with a learned per-channel scale.
#[derive(Module, Debug)]
pub struct RmsNorm {
    weight: Param<Tensor<1>>,
    epsilon: f64,
}

impl RmsNorm {
    /// Creates a layer whose scale starts at one, so it is initially the
    /// identity up to normalization.
    #[must_use]
    pub fn new(hidden_size: usize, epsilon: f64, device: &Device) -> Self {
        Self {
            weight: Initializer::Ones.init([hidden_size], device),
            epsilon,
        }
    }

    /// Epsilon used for numerical stability.
    #[must_use]
    pub const fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Normalizes the last axis of `input`.
    ///
    /// The rank is generic so the same layer serves `[batch, sequence, hidden]`
    /// and a flat `[rows, hidden]`; the normalized axis is always the last one.
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        // `mean_dim` keeps the reduced axis, so the result broadcasts back.
        let mean_square = input.clone().square().mean_dim(D - 1);
        let normalized = input / (mean_square + self.epsilon).sqrt();
        normalized * self.weight.val().unsqueeze::<D>()
    }
}

#[cfg(test)]
mod tests {
    use super::RmsNorm;
    use crate::testing::reference_device;
    use burn::tensor::Tensor;

    /// Scalar reference implementation of the documented formula.
    fn reference(input: &[f32], rows: usize, width: usize, epsilon: f64) -> Vec<f32> {
        let mut out = vec![0.0_f32; input.len()];
        for row in 0..rows {
            let slice = &input[row * width..(row + 1) * width];
            let mean_square: f64 = slice
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                / width as f64;
            let scale = 1.0 / (mean_square + epsilon).sqrt();
            for column in 0..width {
                out[row * width + column] = (f64::from(slice[column]) * scale) as f32;
            }
        }
        out
    }

    fn values(tensor: &Tensor<2>) -> Vec<f32> {
        tensor
            .to_data()
            .try_to_vec::<f32>()
            .expect("the tensor holds f32")
    }

    #[test]
    fn matches_a_scalar_reference_implementation() {
        let device = reference_device();
        let width = 8;
        let rows = 3;
        let epsilon = 1e-5;
        let layer = RmsNorm::new(width, epsilon, &device);

        let input: Vec<f32> = (0..rows * width)
            .map(|index| (index as f32 * 0.37).sin() * 2.0)
            .collect();
        let tensor = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(input.clone(), [rows, width]),
            &device,
        );

        let actual = values(&layer.forward(tensor));
        let expected = reference(&input, rows, width, epsilon);

        for (index, (left, right)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (left - right).abs() < 1e-5,
                "element {index}: {left} vs {right}"
            );
        }
    }

    #[test]
    fn a_constant_row_normalizes_to_one() {
        let device = reference_device();
        let width = 4;
        let layer = RmsNorm::new(width, 1e-5, &device);

        // A row that is constantly 3 has root-mean-square 3, so dividing by it
        // leaves one, and the unit scale leaves it there. This pins the
        // "divide by rms, then scale" order rather than the reverse.
        let tensor = Tensor::<2>::ones([2, width], &device) * 3.0;
        let actual = values(&layer.forward(tensor));
        for value in actual {
            assert!((value - 1.0).abs() < 1e-5, "expected 1.0, got {value}");
        }
    }

    #[test]
    fn the_epsilon_prevents_a_division_by_zero_on_a_zero_row() {
        let device = reference_device();
        let width = 4;
        let layer = RmsNorm::new(width, 1e-5, &device);

        let tensor = Tensor::<2>::zeros([1, width], &device);
        let actual = values(&layer.forward(tensor));

        assert!(
            actual.iter().all(|value| value.is_finite()),
            "a zero row must not produce a non-finite value: {actual:?}"
        );
    }

    #[test]
    fn the_rank_is_generic_and_the_last_axis_is_normalized() {
        let device = reference_device();
        let width = 6;
        let layer = RmsNorm::new(width, 1e-5, &device);

        let flat = layer.forward(Tensor::<2>::ones([5, width], &device));
        assert_eq!(flat.dims(), [5, width]);

        let ranked = layer.forward(Tensor::<3>::ones([2, 5, width], &device));
        assert_eq!(ranked.dims(), [2, 5, width]);
    }

    #[test]
    fn scaling_a_channel_changes_only_that_channel() {
        let device = reference_device();
        let width = 3;
        let mut layer = RmsNorm::new(width, 1e-5, &device);

        // Doubling channel one must double that channel's output.
        let weight = layer.weight.val()
            * Tensor::<1>::from_data(
                burn::tensor::TensorData::new(vec![1.0_f32, 2.0, 1.0], [width]),
                &device,
            );
        layer.weight = burn::module::Param::from_tensor(weight);

        let input = Tensor::<1>::from_data(
            burn::tensor::TensorData::new(vec![1.0_f32, 1.0, 1.0], [width]),
            &device,
        );
        let actual = values(&layer.forward(input.unsqueeze::<2>()));
        // Equal channels normalize to 1/sqrt(1 + eps) each.
        let expected = 1.0 / (1.0_f32 + 1e-5).sqrt();
        assert!((actual[0] - expected).abs() < 1e-6);
        assert!((actual[1] - 2.0 * expected).abs() < 1e-6);
        assert!((actual[2] - expected).abs() < 1e-6);
    }

    #[test]
    fn a_backward_pass_reaches_the_scale() {
        let device = reference_device().autodiff();
        let layer = RmsNorm::new(4, 1e-5, &device);

        let objective = layer.forward(Tensor::<2>::ones([2, 4], &device)).sum();
        let gradients = objective.backward();

        let gradient = layer
            .weight
            .grad(&gradients)
            .expect("the scale participates in the graph");
        crate::grad::ensure_finite(&gradient, crate::error::NumericalQuantity::Gradients)
            .expect("the gradient is finite");
    }
}
