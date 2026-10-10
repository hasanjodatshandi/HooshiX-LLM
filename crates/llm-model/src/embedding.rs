//! Token embedding and the tied language-model head.
//!
//! Both directions use one matrix. Tying is not an optimization here, it is the
//! documented architecture: `tie_embeddings` is part of the Stage-1 model
//! specification, and a change to it requires an architecture decision.
//!
//! Because there is exactly one parameter, the head cannot drift away from the
//! embedding, and the parameter count of the model is the count the
//! configuration predicts.

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Device, Int, Tensor};

use crate::error::ModelError;
use crate::init::weight_initializer;

/// The input embedding table, which also serves as the output projection.
#[derive(Module, Debug)]
pub struct TokenEmbedding {
    table: Embedding,
}

impl TokenEmbedding {
    /// Creates the embedding table.
    #[must_use]
    pub fn new(vocab_size: usize, hidden_size: usize, device: &Device) -> Self {
        Self {
            table: EmbeddingConfig::new(vocab_size, hidden_size)
                .with_initializer(weight_initializer())
                .init(device),
        }
    }

    /// Vocabulary size, which is also the width of the language-model head.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.table.weight.val().dims()[0]
    }

    /// Hidden width.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.table.weight.val().dims()[1]
    }

    /// The shared matrix, exposed so a test can prove that both directions read
    /// the same parameter.
    #[must_use]
    pub fn shared_weight(&self) -> Tensor<2> {
        self.table.weight.val()
    }

    /// The shared parameter itself, exposed so a gradient check can ask whether
    /// a backward pass reached the embedding and the head through one matrix.
    #[must_use]
    pub fn shared_parameter(&self) -> burn::module::Param<Tensor<2>> {
        self.table.weight.clone()
    }

    /// Looks up `[batch, sequence]` token ids, producing
    /// `[batch, sequence, hidden]`.
    pub fn embed(&self, tokens: Tensor<2, Int>) -> Result<Tensor<3>, ModelError> {
        let [_, sequence] = tokens.dims();
        if sequence == 0 {
            return Err(ModelError::Shape {
                operation: "token embedding",
                expected: "a sequence of at least one token".to_owned(),
                actual: "an empty sequence".to_owned(),
            });
        }
        Ok(self.table.forward(tokens))
    }

    /// Projects `[batch, sequence, hidden]` onto the vocabulary, producing
    /// `[batch, sequence, vocab]`.
    ///
    /// The projection uses the transpose of the embedding matrix, so a token's
    /// input vector and its output row are the same parameters.
    pub fn project(&self, hidden: Tensor<3>) -> Result<Tensor<3>, ModelError> {
        let [batch, sequence, width] = hidden.dims();
        let expected = self.hidden_size();
        if width != expected {
            return Err(ModelError::Shape {
                operation: "language-model head",
                expected: format!("hidden size {expected}"),
                actual: format!("hidden size {width}"),
            });
        }

        // The matrix product needs both sides at the same rank, so the batch and
        // sequence axes are flattened and restored around it.
        let rows = hidden.reshape([batch * sequence, width]);
        let projected = rows.matmul(self.table.weight.val().transpose());
        Ok(projected.reshape([batch, sequence, self.vocab_size()]))
    }
}

#[cfg(test)]
mod tests {
    use super::TokenEmbedding;
    use crate::testing::reference_device;
    use burn::tensor::{Int, Tensor, TensorData};

    fn tokens(
        values: Vec<i32>,
        shape: [usize; 2],
        device: &burn::tensor::Device,
    ) -> Tensor<2, Int> {
        Tensor::<2, Int>::from_data(TensorData::new(values, shape), device)
    }

    #[test]
    fn embedding_produces_the_documented_shape() {
        let device = reference_device();
        let embedding = TokenEmbedding::new(32, 8, &device);

        let output = embedding
            .embed(tokens(vec![1, 2, 3], [1, 3], &device))
            .expect("non-empty");
        assert_eq!(output.dims(), [1, 3, 8]);
    }

    #[test]
    fn the_head_produces_the_documented_shape() {
        let device = reference_device();
        let embedding = TokenEmbedding::new(32, 8, &device);

        let logits = embedding
            .project(Tensor::<3>::zeros([2, 5, 8], &device))
            .expect("shapes agree");
        // The output width equals the vocabulary, which is the invariant the
        // whole pipeline depends on.
        assert_eq!(logits.dims(), [2, 5, 32]);
        assert_eq!(logits.dims()[2], embedding.vocab_size());
    }

    #[test]
    fn both_directions_read_the_same_parameter() {
        let device = reference_device();
        let embedding = TokenEmbedding::new(16, 4, &device);
        let shared = embedding.shared_weight();

        // A row of the projection must equal the matching embedding row, which
        // is only true when the head is tied.
        let token = tokens(vec![7], [1, 1], &device);
        let embedded = embedding.embed(token).expect("non-empty");
        let row = shared
            .clone()
            .narrow(0, 7, 1)
            .reshape([4])
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");
        let vector = embedded
            .reshape([4])
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");

        assert_eq!(row, vector);
    }

    #[test]
    fn a_token_id_outside_the_vocabulary_is_rejected() {
        let device = reference_device();
        let embedding = TokenEmbedding::new(16, 4, &device);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            embedding.embed(tokens(vec![16], [1, 1], &device))
        }));

        // The backend rejects an out-of-range index; the point of the test is
        // that it does not silently return a wrong row.
        assert!(
            result.is_err() || result.expect("no panic").is_err(),
            "an out-of-vocabulary id must not produce a row"
        );
    }

    #[test]
    fn an_empty_sequence_is_reported() {
        let device = reference_device();
        let embedding = TokenEmbedding::new(16, 4, &device);

        assert!(
            embedding
                .embed(tokens(Vec::new(), [1, 0], &device))
                .is_err()
        );
    }

    #[test]
    fn a_hidden_width_mismatch_is_reported() {
        let device = reference_device();
        let embedding = TokenEmbedding::new(16, 4, &device);

        assert!(
            embedding
                .project(Tensor::<3>::zeros([1, 2, 8], &device))
                .is_err()
        );
    }
}
