//! The state every request shares.
//!
//! # One model, loaded once
//!
//! The model is held in an `Arc` and never rebuilt. Loading weights per request
//! would turn a millisecond of routing into seconds of disk and allocation, and
//! it is the failure this crate exists to make impossible: there is no code path
//! that constructs a `Transformer` inside a handler.
//!
//! # Readiness is earned, not assumed
//!
//! The server starts **not ready**. Readiness is set only after a generation has
//! actually run, because a model that loaded is not the same as a model that
//! works — a shape error or a backend that cannot execute an operation shows up
//! on the first real forward pass, and traffic must not arrive before that.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use burn::tensor::Device;
use llm_model::Transformer;
use llm_tokenizer::LlmTokenizer;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::limits::ServerLimits;

/// What the handlers share.
pub struct AppState {
    model: Arc<Transformer>,
    tokenizer: Arc<LlmTokenizer>,
    device: Device,
    limits: ServerLimits,
    slots: Arc<Semaphore>,
    ready: AtomicBool,
    eos_token: Option<u32>,
}

impl AppState {
    /// Builds the state around an already-loaded model.
    #[must_use]
    pub fn new(
        model: Transformer,
        tokenizer: LlmTokenizer,
        device: Device,
        limits: ServerLimits,
        eos_token: Option<u32>,
    ) -> Arc<Self> {
        Arc::new(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            device,
            slots: Arc::new(Semaphore::new(limits.max_concurrent_requests)),
            limits,
            ready: AtomicBool::new(false),
            eos_token,
        })
    }

    /// The model every request reads.
    #[must_use]
    pub fn model(&self) -> &Arc<Transformer> {
        &self.model
    }

    /// The tokenizer every request reads.
    #[must_use]
    pub fn tokenizer(&self) -> &Arc<LlmTokenizer> {
        &self.tokenizer
    }

    /// The device the model lives on.
    #[must_use]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// The limits requests are checked against.
    #[must_use]
    pub const fn limits(&self) -> &ServerLimits {
        &self.limits
    }

    /// Token that ends generation, when the model has one.
    #[must_use]
    pub const fn eos_token(&self) -> Option<u32> {
        self.eos_token
    }

    /// Declares the server able to serve.
    ///
    /// Called by the composition root after its self-test, never by a handler.
    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::Release);
    }

    /// Whether the server is able to serve.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Takes a generation slot, or reports that none is free.
    ///
    /// The permit is owned rather than borrowed so it can be moved into the
    /// blocking task that does the work, and released when that task ends —
    /// including when it is cancelled.
    #[must_use]
    pub fn try_take_slot(self: &Arc<Self>) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.slots).try_acquire_owned().ok()
    }
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AppState")
            .field("limits", &self.limits)
            .field("ready", &self.is_ready())
            .field("eos_token", &self.eos_token)
            .finish_non_exhaustive()
    }
}
