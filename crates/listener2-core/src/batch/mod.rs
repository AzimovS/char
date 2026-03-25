mod accumulator;
mod actor;
mod bootstrap;

use std::sync::Arc;

use owhisper_client::{
    ArgmaxAdapter, AssemblyAIAdapter, BatchSttAdapter, DeepgramAdapter, ElevenLabsAdapter,
    FireworksAdapter, GladiaAdapter, HyprnoteAdapter, MistralAdapter, OpenAIAdapter, SonioxAdapter,
};
use tracing::Instrument;

use crate::{BatchEvent, BatchRuntime};

use actor::run_batch_streaming;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, strum::Display, strum::EnumString)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum BatchProvider {
    Argmax,
    Deepgram,
    Soniox,
    AssemblyAI,
    Fireworks,
    OpenAI,
    Gladia,
    ElevenLabs,
    DashScope,
    Mistral,
    Hyprnote,
    Am,
    Cactus,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct BatchParams {
    pub session_id: String,
    pub provider: BatchProvider,
    pub file_path: String,
    #[serde(default)]
    pub model: Option<String>,
    pub base_url: String,
    pub api_key: String,
    #[serde(default)]
    pub languages: Vec<hypr_language::Language>,
    #[serde(default)]
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "snake_case")]
pub enum BatchRunMode {
    Direct,
    Streamed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct BatchRunOutput {
    pub session_id: String,
    pub mode: BatchRunMode,
    pub response: owhisper_interface::batch::Response,
}

pub async fn run_batch(
    runtime: Arc<dyn BatchRuntime>,
    params: BatchParams,
) -> crate::Result<BatchRunOutput> {
    runtime.emit(BatchEvent::BatchStarted {
        session_id: params.session_id.clone(),
    });

    let session_id = params.session_id.clone();
    let result = run_batch_inner(runtime.clone(), params).await;

    if let Err(error) = &result {
        let (code, message) = match error {
            crate::Error::BatchFailed(failure) => (failure.code(), failure.to_string()),
            _ => (crate::BatchErrorCode::Unknown, error.to_string()),
        };

        runtime.emit(BatchEvent::BatchFailed {
            session_id,
            code,
            error: message,
        });
    } else {
        let output = result.as_ref().unwrap();

        runtime.emit(BatchEvent::BatchResponse {
            session_id: output.session_id.clone(),
            response: output.response.clone(),
            mode: output.mode,
        });
        runtime.emit(BatchEvent::BatchCompleted {
            session_id: output.session_id.clone(),
        });
    }

    result
}

const CHUNK_DURATION_MS: u64 = 2 * 60 * 1000;
const CHUNK_THRESHOLD_SECS: f64 = 120.0;

async fn run_batch_inner(
    runtime: Arc<dyn BatchRuntime>,
    params: BatchParams,
) -> crate::Result<BatchRunOutput> {
    let metadata_joined = tokio::task::spawn_blocking({
        let path = params.file_path.clone();
        move || hypr_audio_utils::audio_file_metadata(path)
    })
    .await;

    let metadata_result = match metadata_joined {
        Ok(result) => result,
        Err(err) => {
            let raw_error = format!("{err:?}");
            tracing::error!(error = %raw_error, "audio_metadata_task_join_failed");
            return Err(crate::BatchFailure::AudioMetadataJoinFailed.into());
        }
    };

    let metadata = match metadata_result {
        Ok(metadata) => metadata,
        Err(err) => {
            let raw_error = err.to_string();
            let message = format_user_friendly_error(&raw_error);
            tracing::error!(
                error = %raw_error,
                hyprnote.error.user_message = %message,
                "failed_to_read_audio_metadata"
            );
            return Err(crate::BatchFailure::AudioMetadataReadFailed { message }.into());
        }
    };

    let listen_params = owhisper_interface::ListenParams {
        model: params.model.clone(),
        channels: metadata.channels,
        sample_rate: metadata.sample_rate,
        languages: params.languages.clone(),
        keywords: params.keywords.clone(),
        custom_query: None,
    };

    match params.provider {
        BatchProvider::Am | BatchProvider::Cactus => {
            run_batch_streaming(runtime, params, listen_params).await
        }
        BatchProvider::DashScope => Err(crate::BatchFailure::ProviderRequestFailed {
            message: "DashScope does not support batch transcription".to_string(),
        }
        .into()),
        _ => {
            let provider = params.provider.clone();
            dispatch_batch_simple(&provider, runtime, params, listen_params).await
        }
    }
}

async fn dispatch_batch_simple(
    provider: &BatchProvider,
    runtime: Arc<dyn BatchRuntime>,
    params: BatchParams,
    listen_params: owhisper_interface::ListenParams,
) -> crate::Result<BatchRunOutput> {
    match provider {
        BatchProvider::Argmax => {
            run_batch_simple::<ArgmaxAdapter>(runtime, params, listen_params).await
        }
        BatchProvider::Deepgram => {
            run_batch_simple::<DeepgramAdapter>(runtime, params, listen_params).await
        }
        BatchProvider::Soniox => {
            run_batch_simple::<SonioxAdapter>(runtime, params, listen_params).await
        }
        BatchProvider::AssemblyAI => {
            run_batch_simple::<AssemblyAIAdapter>(runtime, params, listen_params).await
        }
        BatchProvider::Fireworks => {
            run_batch_simple::<FireworksAdapter>(runtime, params, listen_params).await
        }
        BatchProvider::OpenAI => {
            run_batch_simple::<OpenAIAdapter>(runtime, params, listen_params).await
        }
        BatchProvider::Gladia => {
            run_batch_simple::<GladiaAdapter>(runtime, params, listen_params).await
        }
        BatchProvider::ElevenLabs => {
            run_batch_simple::<ElevenLabsAdapter>(runtime, params, listen_params).await
        }
        BatchProvider::Mistral => {
            run_batch_simple::<MistralAdapter>(runtime, params, listen_params).await
        }
        BatchProvider::Hyprnote => {
            run_batch_simple::<HyprnoteAdapter>(runtime, params, listen_params).await
        }
        _ => unreachable!(),
    }
}

async fn run_batch_simple<A: BatchSttAdapter>(
    runtime: Arc<dyn BatchRuntime>,
    params: BatchParams,
    listen_params: owhisper_interface::ListenParams,
) -> crate::Result<BatchRunOutput> {
    let span = session_span(&params.session_id);

    async {
        let duration_result = tokio::task::spawn_blocking({
            let path = params.file_path.clone();
            move || hypr_audio_utils::audio_duration_secs(path)
        })
        .await;

        let audio_duration_secs = match duration_result {
            Ok(Ok(d)) => d,
            _ => 0.0,
        };

        if audio_duration_secs > CHUNK_THRESHOLD_SECS {
            tracing::info!(
                duration_secs = audio_duration_secs,
                "audio exceeds chunk threshold, using chunked transcription"
            );
            return run_batch_chunked::<A>(runtime, params, listen_params).await;
        }

        let client = owhisper_client::BatchClient::<A>::builder()
            .api_base(params.base_url.clone())
            .api_key(params.api_key.clone())
            .params(listen_params)
            .build();

        tracing::debug!("transcribing file: {}", params.file_path);
        let response = match client.transcribe_file(&params.file_path).await {
            Ok(response) => response,
            Err(err) => {
                let raw_error = format!("{err:?}");
                let message = format_user_friendly_error(&raw_error);
                tracing::error!(
                    error = %raw_error,
                    hyprnote.error.user_message = %message,
                    "batch transcription failed"
                );
                return Err(crate::BatchFailure::ProviderRequestFailed { message }.into());
            }
        };
        tracing::info!("batch transcription completed");

        Ok(BatchRunOutput {
            session_id: params.session_id,
            mode: BatchRunMode::Direct,
            response,
        })
    }
    .instrument(span)
    .await
}

const CHUNK_CONCURRENCY: usize = 4;

struct ChunkResult {
    idx: usize,
    offset_secs: f64,
    response: owhisper_interface::batch::Response,
}

async fn run_batch_chunked<A: BatchSttAdapter>(
    runtime: Arc<dyn BatchRuntime>,
    params: BatchParams,
    listen_params: owhisper_interface::ListenParams,
) -> crate::Result<BatchRunOutput> {
    let chunks = tokio::task::spawn_blocking({
        let path = params.file_path.clone();
        move || hypr_audio_utils::chunk_audio_to_wav_files(path, CHUNK_DURATION_MS)
    })
    .await
    .map_err(|err| {
        let message = format!("Failed to chunk audio: {err}");
        tracing::error!(error = %message, "audio_chunk_task_join_failed");
        crate::BatchFailure::ProviderRequestFailed { message }
    })?
    .map_err(|err| {
        let message = format_user_friendly_error(&err.to_string());
        tracing::error!(error = %err, "failed_to_chunk_audio");
        crate::BatchFailure::ProviderRequestFailed { message }
    })?;

    let total_chunks = chunks.len();
    tracing::info!(
        total_chunks,
        concurrency = CHUNK_CONCURRENCY,
        "split audio into chunks for parallel batch transcription"
    );

    if chunks.is_empty() {
        return Ok(BatchRunOutput {
            session_id: params.session_id,
            mode: BatchRunMode::Direct,
            response: owhisper_interface::batch::Response {
                metadata: serde_json::json!({}),
                results: owhisper_interface::batch::Results {
                    channels: vec![owhisper_interface::batch::Channel {
                        alternatives: vec![owhisper_interface::batch::Alternatives {
                            transcript: String::new(),
                            confidence: 0.0,
                            words: Vec::new(),
                        }],
                    }],
                },
            },
        });
    }

    let semaphore = Arc::new(tokio::sync::Semaphore::new(CHUNK_CONCURRENCY));
    let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (cancel_tx, _) = tokio::sync::watch::channel(false);

    let mut handles = Vec::with_capacity(total_chunks);
    for (idx, chunk) in chunks.iter().enumerate() {
        let semaphore = semaphore.clone();
        let completed = completed.clone();
        let runtime = runtime.clone();
        let session_id = params.session_id.clone();
        let offset_secs = chunk.start_offset_secs;
        let chunk_path = chunk.file.path().to_path_buf();
        let mut cancel_rx = cancel_tx.subscribe();
        let client = owhisper_client::BatchClient::<A>::builder()
            .api_base(params.base_url.clone())
            .api_key(params.api_key.clone())
            .params(listen_params.clone())
            .build();

        let handle = tokio::spawn(async move {
            let _permit = semaphore.acquire().await.map_err(|_| {
                crate::Error::from(crate::BatchFailure::ProviderRequestFailed {
                    message: "Chunk semaphore closed".to_string(),
                })
            })?;

            if *cancel_rx.borrow() {
                return Err(crate::Error::from(
                    crate::BatchFailure::ProviderRequestFailed {
                        message: "Batch cancelled".to_string(),
                    },
                ));
            }

            tracing::info!(
                chunk = idx + 1,
                total = total_chunks,
                offset_secs,
                "transcribing chunk"
            );

            let response = tokio::select! {
                result = client.transcribe_file(&chunk_path) => {
                    result.map_err(|err| {
                        let raw_error = format!("{err:?}");
                        let message = format_user_friendly_error(&raw_error);
                        tracing::error!(
                            error = %raw_error,
                            chunk = idx + 1,
                            total = total_chunks,
                            "chunk transcription failed"
                        );
                        crate::Error::from(crate::BatchFailure::ProviderRequestFailed { message })
                    })?
                }
                _ = cancel_rx.changed() => {
                    tracing::info!(chunk = idx + 1, "chunk cancelled");
                    return Err(crate::Error::from(crate::BatchFailure::ProviderRequestFailed {
                        message: "Batch cancelled".to_string(),
                    }));
                }
            };

            let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let percentage = done as f64 / total_chunks as f64;
            runtime.emit(BatchEvent::BatchChunkProgress {
                session_id: session_id.clone(),
                chunk: done,
                total_chunks,
                percentage,
            });

            tracing::info!(
                chunk = idx + 1,
                total = total_chunks,
                done,
                percentage,
                "chunk transcription completed"
            );

            Ok::<ChunkResult, crate::Error>(ChunkResult {
                idx,
                offset_secs,
                response,
            })
        });
        handles.push(handle);
    }

    let mut results = Vec::with_capacity(total_chunks);
    let mut first_error: Option<crate::Error> = None;
    for handle in handles {
        if first_error.is_some() {
            handle.abort();
            continue;
        }
        match handle.await {
            Ok(Ok(chunk_result)) => results.push(chunk_result),
            Ok(Err(err)) => {
                let _ = cancel_tx.send(true);
                first_error = Some(err);
            }
            Err(err) => {
                let _ = cancel_tx.send(true);
                first_error = Some(
                    crate::BatchFailure::ProviderRequestFailed {
                        message: format!("Chunk task panicked: {err}"),
                    }
                    .into(),
                );
            }
        }
    }
    if let Some(err) = first_error {
        return Err(err);
    }

    results.sort_by_key(|r| r.idx);

    let mut all_words: Vec<owhisper_interface::batch::Word> = Vec::new();
    let mut all_transcripts: Vec<String> = Vec::new();
    let mut confidence_sum: f64 = 0.0;
    let mut confidence_count: usize = 0;

    for result in &results {
        for channel in &result.response.results.channels {
            if let Some(alt) = channel.alternatives.first() {
                let transcript = alt.transcript.trim();
                if !transcript.is_empty() {
                    all_transcripts.push(transcript.to_string());
                }

                for word in &alt.words {
                    all_words.push(owhisper_interface::batch::Word {
                        word: word.word.clone(),
                        start: word.start + result.offset_secs,
                        end: word.end + result.offset_secs,
                        confidence: word.confidence,
                        speaker: word.speaker,
                        punctuated_word: word.punctuated_word.clone(),
                    });
                }

                if alt.confidence.is_finite() && alt.confidence > 0.0 {
                    confidence_sum += alt.confidence;
                    confidence_count += 1;
                }
            }
        }
    }

    let avg_confidence = if confidence_count > 0 {
        confidence_sum / confidence_count as f64
    } else {
        0.0
    };

    let merged_response = owhisper_interface::batch::Response {
        metadata: serde_json::json!({ "chunked": true, "total_chunks": total_chunks }),
        results: owhisper_interface::batch::Results {
            channels: vec![owhisper_interface::batch::Channel {
                alternatives: vec![owhisper_interface::batch::Alternatives {
                    transcript: all_transcripts.join(" "),
                    confidence: avg_confidence,
                    words: all_words,
                }],
            }],
        },
    };

    tracing::info!(total_chunks, "chunked batch transcription completed");

    Ok(BatchRunOutput {
        session_id: params.session_id,
        mode: BatchRunMode::Direct,
        response: merged_response,
    })
}

pub(super) fn session_span(session_id: &str) -> tracing::Span {
    tracing::info_span!("session", hyprnote.session.id = %session_id)
}

pub(super) fn format_user_friendly_error(error: &str) -> String {
    let error_lower = error.to_lowercase();

    if error_lower.contains("401") || error_lower.contains("unauthorized") {
        return "Authentication failed. Please check your API key in settings.".to_string();
    }
    if error_lower.contains("403") || error_lower.contains("forbidden") {
        return "Access denied. Your API key may not have permission for this operation."
            .to_string();
    }
    if error_lower.contains("429") || error_lower.contains("rate limit") {
        return "Rate limit exceeded. Please wait a moment and try again.".to_string();
    }
    if error_lower.contains("413")
        || error_lower.contains("too large")
        || error_lower.contains("payload too large")
        || error_lower.contains("entity too large")
    {
        return "The audio file is too large for this provider. Try a shorter recording or check your provider's file size limits.".to_string();
    }
    if error_lower.contains("timeout") || error_lower.contains("timed out") {
        return "Transcription request timed out. The audio file may be too large for this provider, or the server is unresponsive. Try a shorter recording or a different provider.".to_string();
    }
    if error_lower.contains("connection refused")
        || error_lower.contains("failed to connect")
        || error_lower.contains("network")
    {
        return "Could not connect to the transcription service. Please check your internet connection.".to_string();
    }
    if error_lower.contains("invalid audio")
        || error_lower.contains("unsupported format")
        || error_lower.contains("codec")
    {
        return "The audio file format is not supported. Please try a different file.".to_string();
    }
    if error_lower.contains("file not found") || error_lower.contains("no such file") {
        return "Audio file not found. The recording may have been moved or deleted.".to_string();
    }

    error.to_string()
}
