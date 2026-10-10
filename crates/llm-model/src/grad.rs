//! Numerical health checks and the gradient suite.
//!
//! The training loop must fail loudly on a non-finite loss rather than let a
//! `NaN` propagate into the optimizer and be discovered days later. The check
//! lives here, next to the mathematics it guards, so the training crate does not
//! have to reimplement it.
//!
//! # Why the check is arithmetic rather than a comparison
//!
//! The obvious implementation compares each element with itself — `NaN` is the
//! only value that is not equal to itself — and combines the boolean results
//! into one flag that is then read back. That version was written first, and it
//! was wrong in a way that only showed up on a second backend: reading a boolean
//! tensor back is not uniformly supported, and the failure was being folded into
//! "the values are not finite". A backend that could not be *checked* was
//! therefore reported as a backend that had *failed*, and training refused every
//! step on a perfectly healthy loss.
//!
//! What is here instead uses only arithmetic, which every backend supports:
//!
//! - multiplying a finite value by zero gives zero, for either sign and for
//!   subnormals;
//! - multiplying `NaN` by zero gives `NaN`, and multiplying either infinity by
//!   zero gives `NaN`.
//!
//! So the sum of `x * 0` is zero exactly when every element is finite. The sum
//! cannot overflow, because every term is zero. And if the result cannot be read
//! back at all, that is reported as its own error rather than as a failure of
//! the values.

use burn::tensor::Tensor;

use crate::error::{ModelError, NumericalQuantity};

/// Fails if any element of `values` is `NaN` or infinite.
pub fn ensure_finite<const D: usize>(
    values: &Tensor<D>,
    quantity: NumericalQuantity,
) -> Result<(), ModelError> {
    let probe = (values.clone() * 0.0).sum();

    let total = probe
        .to_data()
        .try_to_vec::<f32>()
        .ok()
        .and_then(|values| values.first().copied())
        .ok_or(ModelError::CheckUnavailable { quantity })?;

    if total.is_finite() {
        Ok(())
    } else {
        Err(ModelError::NonFinite { quantity })
    }
}

#[cfg(test)]
mod tests {
    use super::ensure_finite;
    use crate::error::{ModelError, NumericalQuantity};
    use crate::testing::reference_device;
    use burn::tensor::{Tensor, TensorData};

    fn values_tensor(values: Vec<f32>, device: &burn::tensor::Device) -> Tensor<1> {
        let length = values.len();
        Tensor::<1>::from_data(TensorData::new(values, [length]), device)
    }

    #[test]
    fn finite_values_pass() {
        let device = reference_device();
        let values = values_tensor(vec![0.0, 1.0, -1.0, 1e30, -1e30, f32::MIN], &device);
        ensure_finite(&values, NumericalQuantity::Loss).expect("all finite");
    }

    #[test]
    fn a_nan_is_detected() {
        let device = reference_device();
        let values = values_tensor(vec![1.0, f32::NAN, 3.0], &device);
        assert!(matches!(
            ensure_finite(&values, NumericalQuantity::Loss),
            Err(ModelError::NonFinite {
                quantity: NumericalQuantity::Loss
            })
        ));
    }

    #[test]
    fn either_infinity_is_detected() {
        let device = reference_device();
        for special in [f32::INFINITY, f32::NEG_INFINITY] {
            let values = values_tensor(vec![1.0, special], &device);
            assert!(matches!(
                ensure_finite(&values, NumericalQuantity::Gradients),
                Err(ModelError::NonFinite { .. })
            ));
        }
    }

    #[test]
    fn an_empty_tensor_is_finite() {
        let device = reference_device();
        let values = Tensor::<1>::zeros([0], &device);
        ensure_finite(&values, NumericalQuantity::Loss).expect("nothing to be non-finite");
    }

    #[test]
    fn the_reported_quantity_names_what_was_checked() {
        let device = reference_device();
        let values = values_tensor(vec![f32::NAN], &device);

        let error = ensure_finite(&values, NumericalQuantity::Gradients).expect_err("NaN");
        assert!(error.to_string().contains("gradients"), "{error}");
    }

    /// Reference implementation of the scalar function used below.
    fn reference_objective(weights: &[f32], inputs: &[f32]) -> f32 {
        weights
            .iter()
            .zip(inputs.iter())
            .map(|(weight, input)| {
                let product = f64::from(*weight) * f64::from(*input);
                product * product
            })
            .sum::<f64>() as f32
    }

    #[test]
    fn an_autodiff_gradient_matches_a_central_difference() {
        // f(w) = sum((w * x)^2) has the analytic gradient 2 * w * x^2, which a
        // central difference must reproduce. This is the check that catches a
        // wrong derivative, a detached path, or a gradient written into the
        // wrong parameter.
        let device = reference_device().autodiff();
        let inputs = [1.0_f32, 2.0, -0.5, 3.0];
        let initial = [0.2_f32, -0.4, 0.7, 0.1];

        let weights =
            Tensor::<1>::from_data(TensorData::new(initial.to_vec(), [4]), &device).require_grad();
        let input_tensor = Tensor::<1>::from_data(TensorData::new(inputs.to_vec(), [4]), &device);

        let product = weights.clone() * input_tensor;
        let objective = (product.clone() * product).sum();
        let gradients = objective.backward();

        let analytic = weights
            .grad(&gradients)
            .expect("the weights participate in the graph")
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");

        let reference_device = reference_device();
        let epsilon = 1e-3_f32;
        for index in 0..initial.len() {
            let mut above = initial;
            above[index] += epsilon;
            let mut below = initial;
            below[index] -= epsilon;

            let numeric = (reference_objective(&above, &inputs)
                - reference_objective(&below, &inputs))
                / (2.0 * epsilon);

            let difference = (analytic[index] - numeric).abs();
            assert!(
                difference < 1e-3,
                "element {index}: autodiff {} versus central difference {numeric}",
                analytic[index]
            );
        }

        let _ = reference_device;
    }

    #[test]
    fn a_parameter_outside_the_graph_has_no_gradient() {
        // "No accidental detach" has a mirror image: a value that was never
        // connected must not report a gradient either.
        let device = reference_device().autodiff();
        let connected = Tensor::<1>::from_data(TensorData::new(vec![1.0_f32, 2.0], [2]), &device)
            .require_grad();
        let detached = Tensor::<1>::from_data(TensorData::new(vec![1.0_f32, 2.0], [2]), &device);

        let objective = (connected.clone() * connected.clone()).sum();
        let gradients = objective.backward();

        assert!(connected.grad(&gradients).is_some());
        assert!(detached.grad(&gradients).is_none());
    }

    #[test]
    fn gradients_are_non_zero_where_the_objective_depends_on_the_parameter() {
        let device = reference_device().autodiff();
        let parameter = Tensor::<1>::from_data(TensorData::new(vec![0.5_f32, -0.5], [2]), &device)
            .require_grad();
        let input = Tensor::<1>::from_data(TensorData::new(vec![3.0_f32, 3.0], [2]), &device);

        let objective = (parameter.clone() * input).sum();
        let gradients = objective.backward();
        let values = parameter
            .grad(&gradients)
            .expect("in the graph")
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");

        // d/dw sum(w * x) = x, so every entry is three.
        for value in values {
            assert!((value - 3.0).abs() < 1e-6, "expected 3.0, got {value}");
        }
    }
}
