//! Initialization policy.
//!
//! The model specification requires the initialization policy to be centralized,
//! seeded, and free of hidden per-module defaults. Every weight in this crate is
//! therefore drawn from [`weight_initializer`], and a run seeds the device once
//! before construction.
//!
//! Nothing here is tuned. The specification forbids tuning initialization before
//! the tiny-model overfit test works, so the value is a documented starting
//! point rather than a result.
//!
//! # Why construction forces the draw
//!
//! Burn initializes parameters **lazily**: `Initializer::init` builds a
//! `Param::uninitialized` holding a closure, and the values are drawn the first
//! time a parameter is read. Burn's own `num_params` reads `lazy_shape` for this
//! reason, so it does not force anything either.
//!
//! That is a problem for a seeded run. The reference backend keeps one
//! process-wide random source, so the values a parameter receives depend on
//! *when* it is first read: anything that draws in the meantime — another model
//! being used, for instance — advances the source and changes the result. Two
//! models built from the same seed would then differ, and a checkpoint would not
//! be comparable with the run that produced it.
//!
//! [`materialize`] therefore walks the module and reads every parameter
//! immediately after construction. After that call the module holds real
//! tensors, the random source is no longer consulted for it, and the seed alone
//! determines the weights.

use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::Initializer;
use burn::tensor::{Device, Tensor};

/// Standard deviation of the normal distribution every weight is drawn from.
///
/// Small enough that the residual stream does not blow up over twelve blocks,
/// and independent of layer width, which keeps the same value usable at every
/// stage of the roadmap.
pub const WEIGHT_STD: f64 = 0.02;

/// The single initializer every weight matrix and the token embedding use.
#[must_use]
pub fn weight_initializer() -> Initializer {
    Initializer::Normal {
        mean: 0.0,
        std: WEIGHT_STD,
    }
}

/// Seeds the device, so two runs with the same seed initialize identically.
///
/// Called by the composition root before any module is constructed; a model is
/// never responsible for its own seeding, because a module that reseeds would
/// silently reset the run's state.
///
/// Seeding alone is not sufficient on this backend: the draw must also be forced
/// at a known point, which [`materialize`] does. See the module documentation.
pub fn seed(device: &Device, seed: u64) {
    device.seed(seed);
}

/// Forces every float parameter of `module` to be drawn, in field order.
///
/// See the module documentation for why this is necessary. Parameters are
/// visited in the module's own field order, which is fixed by the type, so the
/// order of the draws is deterministic.
pub fn materialize<M: Module>(module: &M) {
    struct Draw;

    impl ModuleVisitor for Draw {
        fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
            // Reading the value is what performs the deferred draw.
            let _ = parameter.val();
        }
    }

    module.visit(&mut Draw);
}
