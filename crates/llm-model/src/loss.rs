//! The causal language-model objective.
//!
//! The objective is next-token prediction: at every position the model is asked
//! for the token that follows, so the target is the input shifted by one.
//!
//! The loss is computed here rather than handed to a framework helper, because
//! the denominator is a correctness property. The specification requires padding
//! to be excluded through an explicit mask, and a batch whose mask ignores
//! everything carries no learning signal at all. That case is reported as
//! [`ModelError::EmptyLossMask`] instead of being allowed to produce a `NaN`
//! that would later be mistaken for numerical instability.

use burn::tensor::{Bool, Int, Tensor};

use crate::error::ModelError;

/// Splits a token sequence into what the model reads and what it must predict.
///
/// For tokens `x0 x1 ... xN` the input is `x0 ... xN-1` and the target is
/// `x1 ... xN`, so the target at a position is the input at the next one.
pub fn next_token_pairs(
    tokens: Tensor<2, Int>,
) -> Result<(Tensor<2, Int>, Tensor<2, Int>), ModelError> {
    let [_, sequence] = tokens.dims();
    if sequence < 2 {
        return Err(ModelError::Shape {
            operation: "next-token shift",
            expected: "a sequence of at least two tokens".to_owned(),
            actual: format!("a sequence of {sequence} token(s)"),
        });
    }

    let input = tokens.clone().narrow(1, 0, sequence - 1);
    let target = tokens.narrow(1, 1, sequence - 1);
    Ok((input, target))
}

/// A mask that counts every position.
///
/// Padding is absent when sequences are packed to a fixed length, so this is the
/// mask a packed batch uses; it keeps the caller's intent explicit rather than
/// leaving the denominator implicit.
#[must_use]
pub fn all_positions(
    batch: usize,
    sequence: usize,
    device: &burn::tensor::Device,
) -> Tensor<2, Bool> {
    Tensor::<2, Bool>::ones([batch, sequence], device)
}

/// Masked cross-entropy over the shifted targets.
///
/// `logits` is `[batch, sequence, vocabulary]`, `targets` is
/// `[batch, sequence]`, and `mask` selects the positions that contribute. The
/// loss is the mean over the selected positions only.
pub fn causal_lm_loss(
    logits: Tensor<3>,
    targets: Tensor<2, Int>,
    mask: Tensor<2, Bool>,
) -> Result<Tensor<1>, ModelError> {
    let [batch, sequence, _] = logits.dims();

    if targets.dims() != [batch, sequence] {
        return Err(ModelError::Shape {
            operation: "causal language-model loss",
            expected: format!("targets of shape [{batch}, {sequence}]"),
            actual: format!("targets of shape {:?}", targets.dims()),
        });
    }
    if mask.dims() != [batch, sequence] {
        return Err(ModelError::Shape {
            operation: "causal language-model loss",
            expected: format!("mask of shape [{batch}, {sequence}]"),
            actual: format!("mask of shape {:?}", mask.dims()),
        });
    }

    // Softmax over the vocabulary axis, which is the last of the three.
    let log_probabilities = burn::tensor::activation::log_softmax(logits, 2);

    // Pick each position's probability for the token it should have predicted.
    let selected = log_probabilities
        .gather(2, targets.unsqueeze_dim::<3>(2))
        .squeeze_dim::<2>(2);

    let weights = mask.float();
    let counted = weights.clone().sum();
    let counted_values = counted.clone().to_data().try_into_vec::<f32>();

    let denominator = match counted_values {
        Ok(values) => values.first().copied().unwrap_or(0.0),
        Err(_) => 0.0,
    };
    // A non-finite or zero denominator means nothing contributed: either the
    // mask is empty or the read-back failed. Both are defects, not instability.
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(ModelError::EmptyLossMask);
    }

    Ok(-(selected * weights).sum() / counted)
}

#[cfg(test)]
mod tests {
    use super::{all_positions, causal_lm_loss, next_token_pairs};
    use crate::error::ModelError;
    use crate::testing::reference_device;
    use burn::tensor::{Bool, Int, Tensor, TensorData};

    fn token_tensor(
        values: Vec<i32>,
        shape: [usize; 2],
        device: &burn::tensor::Device,
    ) -> Tensor<2, Int> {
        Tensor::<2, Int>::from_data(TensorData::new(values, shape), device)
    }

    fn scalar(tensor: Tensor<1>) -> f32 {
        tensor
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32")
            .first()
            .copied()
            .expect("one element")
    }

    #[test]
    fn the_target_is_the_input_shifted_by_one() {
        let device = reference_device();
        let tokens = token_tensor(vec![10, 11, 12, 13], [1, 4], &device);

        let (input, target) = next_token_pairs(tokens).expect("long enough");
        let input = input.to_data().try_to_vec::<i32>().expect("i32");
        let target = target.to_data().try_to_vec::<i32>().expect("i32");

        assert_eq!(input, [10, 11, 12]);
        assert_eq!(target, [11, 12, 13]);
        // The invariant, stated directly: target[t] is input[t + 1].
        for position in 0..input.len() - 1 {
            assert_eq!(target[position], input[position + 1]);
        }
    }

    #[test]
    fn a_sequence_too_short_to_shift_is_reported() {
        let device = reference_device();
        for length in [0, 1] {
            let tokens = token_tensor(vec![1; length], [1, length], &device);
            assert!(matches!(
                next_token_pairs(tokens),
                Err(ModelError::Shape { .. })
            ));
        }
    }

    #[test]
    fn a_uniform_prediction_costs_the_log_of_the_vocabulary() {
        // Equal logits mean a uniform distribution over the vocabulary, so the
        // loss is exactly ln(vocabulary).
        let device = reference_device();
        let vocabulary = 8;
        let logits = Tensor::<3>::zeros([1, 3, vocabulary], &device);
        let targets = token_tensor(vec![0, 1, 2], [1, 3], &device);
        let mask = all_positions(1, 3, &device);

        let loss = scalar(causal_lm_loss(logits, targets, mask).expect("non-empty mask"));
        let expected = (vocabulary as f32).ln();
        assert!((loss - expected).abs() < 1e-5, "{loss} vs {expected}");
    }

    #[test]
    fn a_confident_correct_prediction_costs_almost_nothing() {
        let device = reference_device();
        let vocabulary = 4;
        let mut data = vec![0.0_f32; 2 * vocabulary];
        // Make token 3 overwhelming at both positions.
        data[3] = 30.0;
        data[vocabulary + 3] = 30.0;

        let logits = Tensor::<3>::from_data(TensorData::new(data, [1, 2, vocabulary]), &device);
        let targets = token_tensor(vec![3, 3], [1, 2], &device);
        let mask = all_positions(1, 2, &device);

        let loss = scalar(causal_lm_loss(logits, targets, mask).expect("non-empty mask"));
        assert!(loss < 1e-3, "a confident correct prediction cost {loss}");
    }

    #[test]
    fn a_confident_wrong_prediction_costs_a_lot() {
        let device = reference_device();
        let vocabulary = 4;
        let mut data = vec![0.0_f32; 2 * vocabulary];
        data[0] = 30.0;
        data[vocabulary] = 30.0;

        let logits = Tensor::<3>::from_data(TensorData::new(data, [1, 2, vocabulary]), &device);
        let targets = token_tensor(vec![3, 3], [1, 2], &device);
        let mask = all_positions(1, 2, &device);

        let loss = scalar(causal_lm_loss(logits, targets, mask).expect("non-empty mask"));
        assert!(loss > 20.0, "a confident wrong prediction cost only {loss}");
    }

    #[test]
    fn masked_positions_do_not_contribute_to_the_loss() {
        let device = reference_device();
        let vocabulary = 4;
        // Position 0 predicts correctly and confidently; position 1 is wrong.
        let mut data = vec![0.0_f32; 2 * vocabulary];
        data[0] = 30.0;
        data[vocabulary + 1] = 30.0;

        let logits = Tensor::<3>::from_data(TensorData::new(data, [1, 2, vocabulary]), &device);
        let targets = token_tensor(vec![0, 3], [1, 2], &device);

        let both = all_positions(1, 2, &device);
        let only_first =
            Tensor::<2, Bool>::from_data(TensorData::new(vec![true, false], [1, 2]), &device);

        let with_both =
            scalar(causal_lm_loss(logits.clone(), targets.clone(), both).expect("mask"));
        let with_first = scalar(causal_lm_loss(logits, targets, only_first).expect("mask"));

        // The loss is the mean over the two positions: one is free and one
        // costs about thirty, so the batch costs about fifteen.
        assert!(with_both > 10.0, "the wrong position must cost");
        assert!(
            with_first < 1e-3,
            "ignoring the wrong position must leave almost no loss, got {with_first}"
        );
    }

    #[test]
    fn a_fully_masked_batch_is_reported_rather_than_returned_as_nan() {
        let device = reference_device();
        let logits = Tensor::<3>::zeros([1, 2, 4], &device);
        let targets = token_tensor(vec![0, 1], [1, 2], &device);
        let mask =
            Tensor::<2, Bool>::from_data(TensorData::new(vec![false, false], [1, 2]), &device);

        assert!(matches!(
            causal_lm_loss(logits, targets, mask),
            Err(ModelError::EmptyLossMask)
        ));
    }

    #[test]
    fn a_shape_mismatch_between_logits_and_targets_is_reported() {
        let device = reference_device();
        let logits = Tensor::<3>::zeros([1, 3, 4], &device);
        let targets = token_tensor(vec![0, 1], [1, 2], &device);
        let mask = all_positions(1, 2, &device);

        assert!(matches!(
            causal_lm_loss(logits, targets, mask),
            Err(ModelError::Shape { .. })
        ));
    }

    #[test]
    fn a_mask_that_does_not_match_the_targets_is_reported() {
        let device = reference_device();
        let logits = Tensor::<3>::zeros([1, 3, 4], &device);
        let targets = token_tensor(vec![0, 1, 2], [1, 3], &device);
        let mask = all_positions(1, 2, &device);

        assert!(matches!(
            causal_lm_loss(logits, targets, mask),
            Err(ModelError::Shape { .. })
        ));
    }
}
