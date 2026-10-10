//! Per-layer key and value cache for incremental decoding.
//!
//! Without a cache, generating the *n*-th token re-runs attention over all *n*
//! preceding tokens, so generating a sequence of length *L* costs work
//! proportional to *L*². With a cache, each step attends one new query to the
//! stored keys and values, and the cost is proportional to *L*.
//!
//! The cache stores keys and values **after** rotary rotation and **before**
//! attention. Rotating a key depends only on its own absolute position, which
//! does not change once the token has been seen, so storing the rotated form is
//! correct and saves re-rotating the whole history on every step.
//!
//! # What the cache is not
//!
//! It is not part of the model's parameters and it is not saved in a checkpoint:
//! it is a function of the tokens already fed in, so a resumed run rebuilds it by
//! replaying its prompt.

use burn::tensor::Tensor;

use crate::error::ModelError;

/// One block's stored keys and values, shaped `[batch, heads, positions, head_dim]`.
#[derive(Debug, Clone, Default)]
pub struct LayerCache {
    key: Option<Tensor<4>>,
    value: Option<Tensor<4>>,
}

impl LayerCache {
    /// Creates an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether nothing has been stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.key.is_none()
    }

    /// Number of positions stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.key.as_ref().map_or(0, |key| key.dims()[2])
    }

    /// Drops everything stored.
    pub fn reset(&mut self) {
        self.key = None;
        self.value = None;
    }

    /// Appends one step's keys and values and returns the whole history.
    ///
    /// The returned tensors are what attention must use: the new queries against
    /// the full key and value history. The batch, head and head-dimension sizes
    /// must not change once something has been stored.
    pub fn append(
        &mut self,
        key: Tensor<4>,
        value: Tensor<4>,
    ) -> Result<(Tensor<4>, Tensor<4>), ModelError> {
        if key.dims() != value.dims() {
            return Err(ModelError::Shape {
                operation: "key/value cache append",
                expected: format!(
                    "keys and values of the same shape, got keys {:?}",
                    key.dims()
                ),
                actual: format!("values {:?}", value.dims()),
            });
        }

        let stored = match (&self.key, &self.value) {
            (Some(previous_key), Some(previous_value)) => {
                // A later step may change the sequence length but never the
                // batch, the head count or the head width.
                let [batch, heads, _, head_dim] = previous_key.dims();
                let [new_batch, new_heads, _, new_head_dim] = key.dims();
                if (batch, heads, head_dim) != (new_batch, new_heads, new_head_dim) {
                    return Err(ModelError::Shape {
                        operation: "key/value cache append",
                        expected: format!(
                            "a step shaped [_, {batch}, _, {heads}] by head_dim {head_dim}"
                        ),
                        actual: format!(
                            "[_, {new_batch}, _, {new_heads}] by head_dim {new_head_dim}"
                        ),
                    });
                }

                let key = Tensor::cat(vec![previous_key.clone(), key], 2);
                let value = Tensor::cat(vec![previous_value.clone(), value], 2);
                (key, value)
            }
            _ => (key, value),
        };

        self.key = Some(stored.0.clone());
        self.value = Some(stored.1.clone());
        Ok(stored)
    }
}

/// One [`LayerCache`] per decoder block.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<LayerCache>,
}

impl KvCache {
    /// Creates an empty cache for a model with `num_layers` blocks.
    #[must_use]
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![LayerCache::new(); num_layers],
        }
    }

    /// Number of blocks the cache holds state for.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Whether every layer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(LayerCache::is_empty)
    }

    /// Number of positions stored, read from the first layer.
    ///
    /// Every layer sees the same tokens, so one number describes the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, LayerCache::len)
    }

    /// Mutable access to one layer.
    ///
    /// A model whose depth changed under a cache is a configuration defect, so
    /// an out-of-range index is reported rather than ignored.
    pub fn layer(&mut self, index: usize) -> Result<&mut LayerCache, ModelError> {
        let available = self.layers.len();
        self.layers.get_mut(index).ok_or(ModelError::CacheLayer {
            requested: index,
            available,
        })
    }

    /// Drops everything stored, keeping the layer count.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{KvCache, LayerCache};
    use crate::error::ModelError;
    use crate::testing::reference_device;
    use burn::tensor::{Tensor, TensorData};

    fn step(position: usize, device: &burn::tensor::Device) -> Tensor<4> {
        let values: Vec<f32> = (0..4).map(|index| (position * 10 + index) as f32).collect();
        Tensor::<4>::from_data(TensorData::new(values, [1, 1, 1, 4]), device)
    }

    #[test]
    fn an_empty_cache_stores_nothing() {
        let cache = LayerCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn the_first_append_stores_what_it_was_given() {
        let device = reference_device();
        let mut cache = LayerCache::new();

        let (key, value) = cache
            .append(step(1, &device), step(1, &device))
            .expect("shapes agree");

        assert_eq!(key.dims(), [1, 1, 1, 4]);
        assert_eq!(value.dims(), [1, 1, 1, 4]);
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn appending_extends_the_history_rather_than_replacing_it() {
        let device = reference_device();
        let mut cache = LayerCache::new();

        let _ = cache
            .append(step(1, &device), step(1, &device))
            .expect("ok");
        let _ = cache
            .append(step(2, &device), step(2, &device))
            .expect("ok");
        let (key, _) = cache
            .append(step(3, &device), step(3, &device))
            .expect("ok");

        assert_eq!(cache.len(), 3);
        assert_eq!(key.dims(), [1, 1, 3, 4]);

        // The oldest entry must still be the oldest entry.
        let first = key
            .clone()
            .narrow(2, 0, 1)
            .to_data()
            .try_to_vec::<f32>()
            .expect("f32");
        assert_eq!(first, vec![10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn a_batch_or_head_change_is_rejected() {
        // Sizes that cannot change mid-sequence would silently corrupt the
        // history if they were concatenated anyway.
        let device = reference_device();
        let mut cache = LayerCache::new();
        let _ = cache
            .append(step(1, &device), step(1, &device))
            .expect("ok");

        let wider = Tensor::<4>::zeros([1, 2, 1, 4], &device);
        assert!(matches!(
            cache.append(wider.clone(), wider),
            Err(ModelError::Shape { .. })
        ));
    }

    #[test]
    fn keys_and_values_of_different_shapes_are_rejected() {
        let device = reference_device();
        let mut cache = LayerCache::new();
        assert!(matches!(
            cache.append(step(1, &device), Tensor::<4>::zeros([1, 1, 2, 4], &device)),
            Err(ModelError::Shape { .. })
        ));
    }

    #[test]
    fn resetting_returns_the_cache_to_empty() {
        let device = reference_device();
        let mut cache = LayerCache::new();
        let _ = cache
            .append(step(1, &device), step(1, &device))
            .expect("ok");

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn a_multi_layer_cache_reports_one_length_for_all_layers() {
        let device = reference_device();
        let mut cache = KvCache::new(3);
        assert_eq!(cache.num_layers(), 3);
        assert!(cache.is_empty());

        for index in 0..3 {
            let layer = cache.layer(index).expect("in range");
            let _ = layer
                .append(step(1, &device), step(1, &device))
                .expect("ok");
        }

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn an_out_of_range_layer_is_reported() {
        let mut cache = KvCache::new(2);
        assert!(matches!(
            cache.layer(2),
            Err(ModelError::CacheLayer {
                requested: 2,
                available: 2
            })
        ));
    }

    #[test]
    fn resetting_a_multi_layer_cache_clears_every_layer() {
        let device = reference_device();
        let mut cache = KvCache::new(2);
        for index in 0..2 {
            let layer = cache.layer(index).expect("in range");
            let _ = layer
                .append(step(1, &device), step(1, &device))
                .expect("ok");
        }

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.num_layers(), 2, "reset must not change the depth");
    }
}
