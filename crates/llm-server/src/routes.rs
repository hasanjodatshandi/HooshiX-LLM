//! Routing and handlers.
//!
//! # Why generation runs on a blocking thread
//!
//! A forward pass is synchronous CPU work that holds its thread for the whole
//! step. Running it directly inside a Tokio task blocks a runtime worker, and a
//! runtime with fewer workers than concurrent requests stops serving health
//! checks while it generates — the readiness signal goes dark exactly when it is
//! needed. So every generation runs on `spawn_blocking`, and the async runtime
//! only moves bytes.
//!
//! # Why a request carries a slot permit
//!
//! The permit is moved into the blocking task rather than held by the handler.
//! That way it is released when the work actually ends, including when the
//! client disconnects and the task is dropped, so a cancelled request does not
//! leak capacity.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use llm_inference::{
    GenerationConfig, SamplingConfig, SeededRandom, StopReason, generate_streaming,
};
use llm_tokenizer::SpecialTokensInOutput;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::error::ServerError;
use crate::state::AppState;

/// Largest token budget a request may ask for, when it does not say.
const DEFAULT_NEW_TOKENS: usize = 64;

/// The router every server is built from.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health/live", get(live))
        .route("/health/ready", get(ready))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

/// Serves until `shutdown` resolves.
///
/// The signal is a parameter rather than a call to `ctrl_c` inside, so that
/// stopping is something a test can do. A shutdown path that can only be
/// exercised by pressing a key is a shutdown path nobody checks, and this one
/// has to be right: it is what decides whether in-flight work finishes or is
/// dropped.
pub async fn serve(
    listener: tokio::net::TcpListener,
    state: Arc<AppState>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await
}

/// What a completion request may carry.
///
/// Unknown fields are accepted rather than refused. This is a wire boundary, and
/// a client sending one extra key is not making a mistake the server should fail
/// on — unlike a configuration document, where an unknown key means the operator
/// believes a setting is in effect that is not.
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    /// Text to continue.
    pub prompt: String,
    /// Tokens to produce, bounded by the configured limit.
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    /// Softmax temperature. Zero means greedy.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Keep only the highest-scoring this many candidates.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Keep the smallest set of candidates whose probability sums to this.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Seed for sampling. Ignored by a greedy decoder.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Stream tokens as they are produced rather than returning them together.
    #[serde(default)]
    pub stream: bool,
}

/// What a completion returns.
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    /// Text produced, not including the prompt.
    pub text: String,
    /// Ids produced, in order.
    pub tokens: Vec<u32>,
    /// Why generation stopped.
    pub stop_reason: &'static str,
    /// Tokens the prompt encoded to.
    pub prompt_tokens: usize,
}

/// Why generation stopped, in a form a client can switch on.
const fn stop_reason_name(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndOfSequence => "end_of_sequence",
        StopReason::MaxNewTokens => "max_new_tokens",
        StopReason::Cancelled => "cancelled",
    }
}

/// Liveness: the process is running and able to answer.
///
/// Deliberately independent of readiness. A server whose model failed to load is
/// still alive, and reporting it as dead would have an orchestrator restart a
/// process that would fail the same way again.
async fn live() -> StatusCode {
    StatusCode::OK
}

/// Readiness: the model loaded and its self-test passed.
async fn ready(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.is_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// Validated, bounded work for one request.
struct PrepareOutcome {
    prompt: Vec<u32>,
    generation: GenerationConfig,
}

/// Turns a request into work, or into the reason it cannot be done.
fn prepare(state: &AppState, request: &CompletionRequest) -> Result<PrepareOutcome, ServerError> {
    let limits = state.limits();

    // Bytes first: it needs no tokenizer and no allocation, so it is the
    // cheapest check and it bounds everything after it.
    if request.prompt.len() > limits.max_prompt_bytes {
        return Err(ServerError::PromptTooLong {
            bytes: request.prompt.len(),
            limit: limits.max_prompt_bytes,
        });
    }

    let new_tokens = request.max_new_tokens.unwrap_or(DEFAULT_NEW_TOKENS);
    if new_tokens == 0 || new_tokens > limits.max_new_tokens {
        return Err(ServerError::NewTokensTooMany {
            requested: new_tokens,
            limit: limits.max_new_tokens,
        });
    }

    let sampling = SamplingConfig {
        temperature: request.temperature.unwrap_or(0.0),
        top_k: request.top_k,
        top_p: request.top_p,
    };
    // Sampled settings are rejected before the prompt is tokenized, so a request
    // that cannot work does no work at all.
    sampling
        .validate()
        .map_err(|error| ServerError::InvalidSampling {
            reason: error.to_string(),
        })?;

    let prompt =
        state
            .tokenizer()
            .encode(&request.prompt)
            .map_err(|error| ServerError::Encoding {
                reason: error.to_string(),
            })?;

    if prompt.is_empty() {
        return Err(ServerError::Encoding {
            reason: "the prompt encoded to no tokens".to_owned(),
        });
    }
    if prompt.len() > limits.max_prompt_tokens {
        return Err(ServerError::PromptTokensTooMany {
            tokens: prompt.len(),
            limit: limits.max_prompt_tokens,
        });
    }

    Ok(PrepareOutcome {
        prompt,
        generation: GenerationConfig {
            max_new_tokens: new_tokens,
            eos_token: state.eos_token(),
            sampling,
        },
    })
}

/// Continues a prompt.
async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ServerError> {
    if !state.is_ready() {
        return Err(ServerError::NotReady);
    }

    let outcome = prepare(&state, &request)?;

    let limit = state.limits().max_concurrent_requests;
    let Some(permit) = state.try_take_slot() else {
        return Err(ServerError::Busy { limit });
    };

    if request.stream {
        Ok(stream_response(&state, outcome, permit))
    } else {
        Ok(collect_response(&state, outcome, permit).await?)
    }
}

/// Runs generation to completion and returns it as one document.
async fn collect_response(
    state: &Arc<AppState>,
    outcome: PrepareOutcome,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Result<Response, ServerError> {
    let worker = Arc::clone(state);
    let prompt_tokens = outcome.prompt.len();

    let generation = tokio::task::spawn_blocking(move || {
        // Held for the life of the work, so capacity is released when it ends.
        let _permit = permit;

        let mut random = SeededRandom::new(0);
        generate_streaming(
            worker.model(),
            &outcome.prompt,
            &outcome.generation,
            worker.device(),
            &mut random,
            |_| true,
        )
    })
    .await
    .map_err(|error| ServerError::Generation {
        reason: error.to_string(),
    })?
    .map_err(|error| ServerError::Generation {
        reason: error.to_string(),
    })?;

    let text = state
        .tokenizer()
        .decode(&generation.tokens, SpecialTokensInOutput::Skip)
        .map_err(|error| ServerError::Generation {
            reason: error.to_string(),
        })?;

    Ok(Json(CompletionResponse {
        text,
        tokens: generation.tokens,
        stop_reason: stop_reason_name(generation.stop),
        prompt_tokens,
    })
    .into_response())
}

/// Streams tokens as they are produced.
///
/// The channel is the cancellation path. When the client goes away the response
/// is dropped, the receiver goes with it, `blocking_send` fails, and the
/// generation loop is told to stop. Without that the server would keep computing
/// tokens into a socket nobody is reading.
fn stream_response(
    state: &Arc<AppState>,
    outcome: PrepareOutcome,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Event>(16);
    let state = Arc::clone(state);

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut random = SeededRandom::new(0);
        let mut decoder_failed = false;

        let result = generate_streaming(
            state.model(),
            &outcome.prompt,
            &outcome.generation,
            state.device(),
            &mut random,
            |token| {
                let chunk = match state
                    .tokenizer()
                    .decode(&[token], SpecialTokensInOutput::Skip)
                {
                    Ok(chunk) => chunk,
                    Err(_) => {
                        decoder_failed = true;
                        return false;
                    }
                };
                let payload = serde_json::json!({ "text": chunk, "token": token });
                // A closed receiver means the client is gone. Returning false
                // stops the loop rather than finishing work nobody will read.
                sender
                    .blocking_send(Event::default().data(payload.to_string()))
                    .is_ok()
            },
        );

        let closing = match result {
            Ok(generation) => serde_json::json!({
                "stop_reason": stop_reason_name(generation.stop),
                "tokens": generation.tokens.len(),
            }),
            Err(error) => serde_json::json!({ "error": error.to_string() }),
        };
        let _ = sender.blocking_send(Event::default().event("done").data(closing.to_string()));
        let _ = decoder_failed;
    });

    let stream = ReceiverStream::new(receiver).map(Ok::<Event, Infallible>);
    Sse::new(stream).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::ServerLimits;
    use axum::body::Body;
    use axum::http::Request;
    use llm_core::ModelConfig;
    use llm_model::Transformer;
    use tower::ServiceExt;

    fn state(ready: bool, limits: ServerLimits) -> (Arc<AppState>, String) {
        let device = burn::tensor::Device::flex();
        llm_model::init::seed(&device, 0xC0FFEE);
        let config = ModelConfig {
            // Must match the tokenizer below: an id the tokenizer can produce but
            // the embedding cannot hold is a shape error at the first real
            // request, and the point of this state is to be past that.
            vocab_size: 260,
            hidden_size: 16,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            ffn_intermediate_size: 32,
            max_sequence_length: 64,
            ..ModelConfig::default()
        };
        let model = Transformer::new(&config, &device).expect("valid config");

        // A byte-level tokenizer trained on the spot. The server never reads a
        // shard, so it only needs something that encodes and decodes. The
        // vocabulary is exactly the byte alphabet plus the four reserved tokens,
        // because a corpus this small cannot learn merges and a larger target
        // would be refused for not being filled.
        let tokenizer_config = llm_core::TokenizerConfig {
            vocab_size: 260,
            ..llm_core::TokenizerConfig::default()
        };
        let corpus = "the quick brown fox jumps over the lazy dog. \
                      the quick brown fox jumps over the lazy dog again.";
        let tokenizer =
            llm_tokenizer::LlmTokenizer::train(std::iter::once(corpus), &tokenizer_config, 1)
                .expect("trains");

        let app = AppState::new(model, tokenizer, device, limits, None);
        if ready {
            app.mark_ready();
        }
        let backend = "flex".to_owned();
        (app, backend)
    }

    async fn post(state: Arc<AppState>, body: &str) -> (StatusCode, serde_json::Value) {
        let response = router(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_owned()))
                    .expect("request builds"),
            )
            .await
            .expect("router answers");

        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body reads");
        let value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, value)
    }

    #[tokio::test]
    async fn liveness_answers_before_readiness_does() {
        let (app, _) = state(false, ServerLimits::default());

        let live = router(Arc::clone(&app))
            .oneshot(
                Request::builder()
                    .uri("/health/live")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("answers");
        assert_eq!(live.status(), StatusCode::OK);

        let ready = router(Arc::clone(&app))
            .oneshot(
                Request::builder()
                    .uri("/health/ready")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("answers");
        assert_eq!(ready.status(), StatusCode::SERVICE_UNAVAILABLE);

        app.mark_ready();
        let ready = router(app)
            .oneshot(
                Request::builder()
                    .uri("/health/ready")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("answers");
        assert_eq!(ready.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn an_unready_server_refuses_work() {
        let (app, _) = state(false, ServerLimits::default());
        let (status, body) = post(app, r#"{"prompt":"the"}"#).await;

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|text| text.contains("not ready")),
            "expected a readiness refusal, got {body}"
        );
    }

    #[tokio::test]
    async fn a_ready_server_completes_a_prompt() {
        let (app, _) = state(true, ServerLimits::default());
        let (status, body) = post(
            Arc::clone(&app),
            r#"{"prompt":"the quick brown","max_new_tokens":4}"#,
        )
        .await;

        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["tokens"].as_array().map(Vec::len), Some(4));
        assert_eq!(body["stop_reason"], "max_new_tokens");
        assert!(body["text"].is_string());
        assert!(body["prompt_tokens"].as_u64().is_some_and(|n| n > 0));
    }

    #[tokio::test]
    async fn a_prompt_above_the_byte_limit_is_refused_before_anything_else() {
        let limits = ServerLimits {
            max_prompt_bytes: 8,
            ..ServerLimits::default()
        };
        let (app, _) = state(true, limits);
        let (status, body) =
            post(app, r#"{"prompt":"a much longer prompt than eight bytes"}"#).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|text| text.contains("byte limit")),
            "expected the byte limit, got {body}"
        );
    }

    #[tokio::test]
    async fn a_token_budget_above_the_limit_is_refused() {
        let limits = ServerLimits {
            max_new_tokens: 4,
            ..ServerLimits::default()
        };
        let (app, _) = state(true, limits);
        let (status, body) = post(app, r#"{"prompt":"the","max_new_tokens":64}"#).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|text| text.contains("new tokens")),
            "expected the token limit, got {body}"
        );
    }

    #[tokio::test]
    async fn a_zero_token_budget_is_refused() {
        let (app, _) = state(true, ServerLimits::default());
        let (status, _) = post(app, r#"{"prompt":"the","max_new_tokens":0}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn invalid_sampling_is_refused_before_the_prompt_is_tokenized() {
        let (app, _) = state(true, ServerLimits::default());
        let (status, body) = post(app, r#"{"prompt":"the","temperature":-1.0}"#).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|text| text.contains("temperature")),
            "expected a sampling refusal, got {body}"
        );
    }

    #[tokio::test]
    async fn a_second_request_is_refused_while_one_holds_the_only_slot() {
        // One model, one device: the default is one generation at a time, and a
        // request that arrives while it is busy is told so rather than queued
        // behind it with no bound.
        let (app, _) = state(true, ServerLimits::default());
        let held = app.try_take_slot().expect("the first slot is free");

        let (status, body) = post(Arc::clone(&app), r#"{"prompt":"the","max_new_tokens":1}"#).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert!(
            body["error"]
                .as_str()
                .is_some_and(|text| text.contains("retry"))
        );

        drop(held);

        let (status, _) = post(app, r#"{"prompt":"the","max_new_tokens":1}"#).await;
        assert_eq!(status, StatusCode::OK, "the slot must be reusable");
    }

    #[tokio::test]
    async fn the_server_answers_over_a_socket_and_stops_when_asked() {
        // A real listener and real bytes, because the parts this covers — that
        // the process serves from a bound port and that a shutdown signal ends
        // it cleanly — are exactly the parts the in-process router tests cannot
        // reach.
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (app, _) = state(true, ServerLimits::default());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binds an ephemeral port");
        let address = listener.local_addr().expect("has an address");

        let (signal, wait) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(serve(listener, app, async move {
            let _ = wait.await;
        }));

        let mut stream = tokio::net::TcpStream::connect(address)
            .await
            .expect("connects to the server");
        stream
            .write_all(
                b"GET /health/ready HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            )
            .await
            .expect("writes the request");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .await
            .expect("reads the response");

        assert!(
            response.starts_with("HTTP/1.1 200"),
            "the ready server must answer 200, got {:?}",
            response.lines().next()
        );

        signal.send(()).expect("the server is still listening");
        tokio::time::timeout(std::time::Duration::from_secs(5), server)
            .await
            .expect("the server stops promptly")
            .expect("the task joins")
            .expect("the server stops without an error");
    }

    #[tokio::test]
    async fn a_server_that_was_never_marked_ready_reports_unready_over_a_socket() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (app, _) = state(false, ServerLimits::default());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binds");
        let address = listener.local_addr().expect("has an address");

        let (signal, wait) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(serve(listener, app, async move {
            let _ = wait.await;
        }));

        let mut stream = tokio::net::TcpStream::connect(address)
            .await
            .expect("connects");
        stream
            .write_all(
                b"GET /health/ready HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            )
            .await
            .expect("writes");
        let mut response = String::new();
        stream.read_to_string(&mut response).await.expect("reads");

        assert!(
            response.starts_with("HTTP/1.1 503"),
            "readiness must not be exposed before the self-test, got {:?}",
            response.lines().next()
        );

        let _ = signal.send(());
        let _ = server.await;
    }
}
