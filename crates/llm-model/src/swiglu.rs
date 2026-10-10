//! The gated feed-forward network.
//!
//! Reference form, from the model specification:
//!
//! ```text
//! SwiGLU(x) = SiLU(x W_gate) * (x W_up)
//! FFN(x)    = SwiGLU(x) W_down
//! ```
//!
//! Three matrices rather than the two of a plain feed-forward network. The
//! hidden width is therefore chosen so that the parameter count matches a plain
//! network of the documented intermediate size.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::{Device, Tensor};

use crate::error::ModelError;

/// A gated feed-forward network.
#[derive(Module, Debug)]
pub struct SwiGlu {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl SwiGlu {
    /// Creates the network.
    #[must_use]
    pub fn new(hidden_size: usize, intermediate_size: usize, bias: bool, device: &Device) -> Self {
        let project =
            |from: usize, to: usize| LinearConfig::new(from, to).with_bias(bias).init(device);
        Self {
            gate: project(hidden_size, intermediate_size),
            up: project(hidden_size, intermediate_size),
            down: project(intermediate_size, hidden_size),
        }
    }

    /// Applies the network to `[batch, sequence, hidden]`.
    pub fn forward(&self, input: Tensor<3>) -> Result<Tensor<3>, ModelError> {
        let [_, _, hidden] = input.dims();
        let expected = self.down.weight.val().dims()[1];
        if hidden != expected {
            return Err(ModelError::Shape {
                operation: "gated feed-forward network",
                expected: format!("hidden size {expected}"),
                actual: format!("hidden size {hidden}"),
            });
        }

        let gated = silu(self.gate.forward(input.clone())) * self.up.forward(input);
        Ok(self.down.forward(gated))
    }
}

#[cfg(test)]
mod tests {
    use super::SwiGlu;
    use crate::testing::reference_device;
    use burn::tensor::{Tensor, TensorData};

    fn values(tensor: &Tensor<3>) -> Vec<f32> {
        tensor
            .to_data()
            .try_to_vec::<f32>()
            .expect("the tensor holds f32")
    }

    fn input(device: &burn::tensor::Device) -> Tensor<3> {
        let data: Vec<f32> = (0..2 * 3 * 4)
            .map(|index| (index as f32 * 0.31).sin())
            .collect();
        Tensor::<3>::from_data(TensorData::new(data, [2, 3, 4]), device)
    }

    #[test]
    fn the_output_shape_matches_the_input_shape() {
        let device = reference_device();
        let network = SwiGlu::new(4, 9, false, &device);
        let output = network.forward(input(&device)).expect("shapes agree");
        assert_eq!(output.dims(), [2, 3, 4]);
    }

    #[test]
    fn the_output_is_finite_for_a_range_of_inputs() {
        let device = reference_device();
        let network = SwiGlu::new(4, 16, false, &device);

        for scale in [0.0_f32, 1.0, 100.0, -100.0] {
            let output = network
                .forward(input(&device) * scale)
                .expect("shapes agree");
            assert!(
                values(&output).iter().all(|value| value.is_finite()),
                "scale {scale} produced a non-finite value"
            );
        }
    }

    #[test]
    fn a_zero_gate_makes_the_branch_contribute_nothing() {
        // SiLU(0) is zero, so zeroing the gate weights silences the up branch
        // and the whole network collapses to its bias, which is absent here.
        let device = reference_device();
        let mut network = SwiGlu::new(4, 6, false, &device);
        // A Burn `Linear` weight is `[input, output]`, so the gate is `[4, 6]`.
        network.gate.weight = burn::module::Param::from_tensor(Tensor::<2>::zeros([4, 6], &device));

        let output = values(&network.forward(input(&device)).expect("shapes agree"));
        assert!(
            output.iter().all(|value| value.abs() < 1e-6),
            "a silenced gate must silence the branch: {output:?}"
        );
    }

    #[test]
    fn the_gate_and_the_up_projection_are_independent_matrices() {
        let device = reference_device();
        let network = SwiGlu::new(4, 6, false, &device);

        let gate = network
            .gate
            .weight
            .val()
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");
        let up = network
            .up
            .weight
            .val()
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");
        assert_ne!(gate, up, "gate and up must not share one matrix");
    }

    #[test]
    fn a_hidden_size_mismatch_is_reported() {
        let device = reference_device();
        let network = SwiGlu::new(4, 6, false, &device);
        let wrong = Tensor::<3>::zeros([1, 2, 8], &device);

        assert!(network.forward(wrong).is_err());
    }

    #[test]
    fn a_backward_pass_reaches_all_three_matrices() {
        let device = reference_device().autodiff();
        let network = SwiGlu::new(4, 6, false, &device);

        let objective = network.forward(input(&device)).expect("shapes agree").sum();
        let gradients = objective.backward();

        for (name, parameter) in [
            ("gate", &network.gate.weight),
            ("up", &network.up.weight),
            ("down", &network.down.weight),
        ] {
            let gradient = parameter
                .grad(&gradients)
                .unwrap_or_else(|| panic!("the {name} matrix must be in the graph"));
            crate::grad::ensure_finite(&gradient, crate::error::NumericalQuantity::Gradients)
                .unwrap_or_else(|error| panic!("the {name} gradient is not finite: {error}"));
        }
    }
}
