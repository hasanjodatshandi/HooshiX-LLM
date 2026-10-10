//! The causal attention mask.
//!
//! The mask is the single invariant that makes autoregressive training honest: a
//! token at position `t` must never observe a position greater than `t`. If it
//! does, the model can read the answer it is being asked to predict, the loss
//! falls for the wrong reason, and nothing downstream detects it.
//!
//! The mask is built as a boolean "should be masked" tensor so it can be handed
//! to `masked_fill` directly, and it is built on the device it will be used on.

use burn::tensor::{Bool, Device, Tensor};

/// Builds the mask for a query block attending to a key block.
///
/// The returned tensor has shape `[1, 1, query_length, key_length]`, so it
/// broadcasts across batch and heads. `true` marks a position that must be
/// masked out.
///
/// `position_offset` is the absolute position of the first query. It is
/// `key_length - query_length` during cached decoding, where one new query
/// attends to a longer key history, and zero for a full sequence.
///
/// # Panics
///
/// Does not panic: an inconsistent offset is clamped rather than allowed to
/// produce an out-of-range index.
#[must_use]
pub fn causal_mask(
    query_length: usize,
    key_length: usize,
    position_offset: usize,
    device: &Device,
) -> Tensor<4, Bool> {
    // A query at absolute position p must see keys with absolute position <= p.
    // With the query block starting at `position_offset`, that is a lower
    // triangular mask shifted by the offset.
    let offset = i64::try_from(position_offset).unwrap_or(i64::MAX);
    Tensor::<2, Bool>::tril_mask([query_length, key_length], offset, device)
        .unsqueeze::<3>()
        .unsqueeze::<4>()
}

#[cfg(test)]
mod tests {
    use super::causal_mask;
    use crate::testing::reference_device;
    use burn::tensor::Tensor;

    fn mask_rows(query_length: usize, key_length: usize, offset: usize) -> Vec<Vec<bool>> {
        let device = reference_device();
        let mask = causal_mask(query_length, key_length, offset, &device);
        let flat = mask
            .reshape([query_length, key_length])
            .to_data()
            .try_to_vec::<bool>()
            .expect("a boolean mask");

        (0..query_length)
            .map(|row| flat[row * key_length..(row + 1) * key_length].to_vec())
            .collect()
    }

    #[test]
    fn a_query_never_sees_a_future_key() {
        for length in 1..6 {
            let rows = mask_rows(length, length, 0);
            for (position, row) in rows.iter().enumerate() {
                for (key, masked) in row.iter().enumerate() {
                    let should_be_masked = key > position;
                    assert_eq!(
                        *masked, should_be_masked,
                        "length {length}: query {position} against key {key}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_first_query_sees_only_the_first_key() {
        let rows = mask_rows(4, 4, 0);
        assert_eq!(rows[0], [false, true, true, true]);
    }

    #[test]
    fn the_last_query_sees_every_key() {
        let rows = mask_rows(4, 4, 0);
        assert_eq!(rows[3], [false, false, false, false]);
    }

    #[test]
    fn a_shifted_block_sees_the_history_it_should() {
        // One new query at absolute position 3, attending to four keys at
        // absolute positions 0..3. Only the key at position 4 would be future,
        // and there is none, so nothing is masked.
        let rows = mask_rows(1, 4, 3);
        assert_eq!(rows[0], [false, false, false, false]);

        // A query at absolute position 1, with four keys 0..3, must not see
        // keys 2 and 3.
        let rows = mask_rows(1, 4, 1);
        assert_eq!(rows[0], [false, false, true, true]);
    }

    #[test]
    fn the_mask_broadcasts_across_batch_and_heads() {
        let device = reference_device();
        let mask = causal_mask(3, 3, 0, &device);
        assert_eq!(mask.dims(), [1, 1, 3, 3]);

        // Filling a ranked score tensor must leave its shape alone.
        let scores = Tensor::<4>::zeros([2, 5, 3, 3], &device);
        let masked = scores.mask_fill(mask, f32::NEG_INFINITY);
        assert_eq!(masked.dims(), [2, 5, 3, 3]);
    }

    #[test]
    fn masked_positions_become_negative_infinity_before_softmax() {
        let device = reference_device();
        let scores = Tensor::<4>::zeros([1, 1, 2, 2], &device);
        let masked = scores.mask_fill(causal_mask(2, 2, 0, &device), f32::NEG_INFINITY);
        let weights = burn::tensor::activation::softmax(masked, 3)
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");

        // Row 0 can only attend to key 0, so that weight is one and the other
        // is zero; row 1 sees both equally.
        assert!((weights[0] - 1.0).abs() < 1e-6, "{weights:?}");
        assert!(weights[1].abs() < 1e-6, "{weights:?}");
        assert!((weights[2] - 0.5).abs() < 1e-6, "{weights:?}");
        assert!((weights[3] - 0.5).abs() < 1e-6, "{weights:?}");
    }
}
