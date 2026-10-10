//! Training-loop orchestration for the Rust-native LLM stack.
//!
//! This crate owns the loop: batches go in, a forward pass and an objective
//! produce a loss, a backward pass produces gradients, and an optimizer produces
//! new parameters. It owns no model mathematics, no dataset parsing and no
//! checkpoint format. The format itself lives in `llm-checkpoint`, because the
//! inference side reads it too.
//!
//! # What is implemented
//!
//! AdamW, a closed-form warmup-then-cosine schedule, norm gradient clipping,
//! loss-summing accumulation over micro-batches, validation evaluation, token
//! accounting and a metrics record.
//!
//! # What is not
//!
//! Reduced-precision training. The specification allows BF16 only after the
//! FP32 path is numerically stable and a forward/backward parity test passes on
//! the target backend, and no accelerator is present to run either. The Stage-1
//! default is therefore FP32, as the model specification requires.

#![deny(unsafe_code)]

pub use llm_checkpoint as checkpoint;

pub mod error;
pub mod metrics;
pub mod scheduler;
pub mod trainer;

pub use error::{NonFiniteLoss, TrainingError};
pub use llm_checkpoint::{
    CHECKPOINT_FORMAT_VERSION, CheckpointError, CheckpointExpectation, CheckpointMetadata,
    DataCursor,
};
pub use metrics::{MetricsLog, StepMetrics};
pub use scheduler::WarmupCosineSchedule;
pub use trainer::{TokenBatch, Trainer, ValidationSchedule, evaluate};
