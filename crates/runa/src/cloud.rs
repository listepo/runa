//! Cloud backend routing for `runa run` (P3.7).

use std::io::{self, Write};

use std::path::Path;

use runa_cloud::openai::{
    AudioFormat, AudioInput, ChatMessage, ChatRequest, CloudEvent as OpenAiEvent, OpenAiClient,
};
use runa_cloud::{
    AnthropicClient, AnthropicEvent, AnthropicRequest, ChatTurn, CloudRef, PriceTable, Provider,
    parse_cloud_ref, resolve_api_key,
};
use runa_core::ThinkConfig;

pub fn run_cloud(
    cloud: &CloudRef,
    prompt: &str,
    think: ThinkConfig,
    max_tokens: u32,
    json: bool,
    audio: Option<&Path>,
    audio_pref: runa_media::AudioRoutePref,
) -> Result<(), String> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    rt.block_on(run_cloud_async(
        cloud, prompt, think, max_tokens, json, audio, audio_pref,
    ))
}

async fn run_cloud_async(
    cloud: &CloudRef,
    prompt: &str,
    think: ThinkConfig,
    max_tokens: u32,
    json: bool,
    audio: Option<&Path>,
    audio_pref: runa_media::AudioRoutePref,
) -> Result<(), String> {
    let (prompt, oai_audio) = prepare_cloud_audio(cloud, prompt, audio, audio_pref)?;
    let prices = PriceTable::load();
    let key = resolve_api_key(cloud.provider).map_err(|e| e.to_string())?;
    eprintln!(
        "cloud: {}:{} (key {})",
        cloud.provider_name(),
        cloud.model,
        key.redacted()
    );

    let (text, reasoning, input_tok, output_tok) = match cloud.provider {
        Provider::OpenAi => {
            let base = std::env::var("OPENAI_BASE_URL").ok();
            let client = OpenAiClient::new(key.value, base.as_deref());
            let req = ChatRequest {
                model: cloud.model.clone(),
                messages: vec![{
                    let mut m = ChatMessage::user(prompt);
                    m.audio = oai_audio;
                    m
                }],
                think,
                max_tokens: Some(max_tokens),
            };
            let events = client.complete(req).await.map_err(|e| e.to_string())?;
            drain_openai(events)?
        }
        Provider::Anthropic => {
            let _ = oai_audio;
            let client = AnthropicClient::new(key.value);
            let req = AnthropicRequest {
                model: cloud.model.clone(),
                system: None,
                messages: vec![ChatTurn {
                    role: "user".into(),
                    text: prompt.to_string(),
                }],
                think,
                max_tokens,
                images: vec![],
                pdfs: vec![],
                stream: false,
            };
            let events = client.generate(&req).await?;
            drain_anthropic(events)?
        }
    };

    if json {
        println!(
            "{{\"text\":\"{}\",\"reasoning\":\"{}\",\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{}}}}}",
            crate::json_escape(&text),
            crate::json_escape(&reasoning),
            input_tok,
            output_tok
        );
    } else {
        if think.show && !reasoning.is_empty() {
            eprint!("{reasoning}");
            io::stderr().flush().map_err(|e| format!("stderr: {e}"))?;
        }
        print!("{text}");
        io::stdout().flush().map_err(|e| format!("stdout: {e}"))?;
        println!();
        eprintln!("tokens: {input_tok} in / {output_tok} out");
        if let Some(line) =
            prices.cost_line(cloud.provider_name(), &cloud.model, input_tok, output_tok)
        {
            eprintln!("{line}");
        }
    }
    Ok(())
}

fn prepare_cloud_audio(
    cloud: &CloudRef,
    prompt: &str,
    audio: Option<&Path>,
    pref: runa_media::AudioRoutePref,
) -> Result<(String, Vec<AudioInput>), String> {
    let Some(path) = audio else {
        return Ok((prompt.to_string(), Vec::new()));
    };
    let backend = match cloud.provider {
        Provider::OpenAi => runa_media::AudioBackend::OpenAi {
            audio_capable: runa_media::openai_audio_capable(&cloud.model),
        },
        Provider::Anthropic => runa_media::AudioBackend::Anthropic,
    };
    match runa_media::select_audio_route(pref, backend)? {
        runa_media::AudioPlan::Transcribe => Ok((fold_audio_transcript(prompt, path)?, Vec::new())),
        runa_media::AudioPlan::OpenAiInputAudio => {
            Ok((prompt.to_string(), vec![openai_wav_input(path)?]))
        }
        runa_media::AudioPlan::Native => {
            Err("native audio is local-only; cloud uses input_audio or transcript".into())
        }
    }
}

pub(crate) fn fold_audio_transcript(prompt: &str, audio: &Path) -> Result<String, String> {
    let t = runa_media::transcribe_file(audio, &runa_media::AsrOptions::default())
        .map_err(|e| e.to_string())?;
    let text = t.text.trim();
    if text.is_empty() {
        return Err("asr produced empty transcript".into());
    }
    Ok(format!("{prompt}\n\n[audio transcript]\n{text}"))
}

fn openai_wav_input(audio: &Path) -> Result<AudioInput, String> {
    let decoded = runa_media::decode_audio(audio).map_err(|e| e.to_string())?;
    let wav = runa_media::pcm_to_wav_bytes(&decoded.samples, decoded.probe.pcm_sample_rate)
        .map_err(|e| e.to_string())?;
    Ok(AudioInput {
        data_b64: runa_media::wav_base64(&wav),
        format: AudioFormat::Wav,
    })
}

pub fn cloud_from_on_unfit(spec: &str) -> Result<CloudRef, String> {
    parse_cloud_ref(spec).ok_or_else(|| format!("on_unfit cloud:{spec}: need backend:model"))
}

fn drain_openai(events: Vec<OpenAiEvent>) -> Result<(String, String, u32, u32), String> {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut in_tok = 0u32;
    let mut out_tok = 0u32;
    for ev in events {
        match ev {
            OpenAiEvent::Text(s) => text.push_str(&s),
            OpenAiEvent::Reasoning(s) => reasoning.push_str(&s),
            OpenAiEvent::Usage {
                prompt_tokens,
                completion_tokens,
            } => {
                in_tok = prompt_tokens;
                out_tok = completion_tokens;
            }
        }
    }
    Ok((text, reasoning, in_tok, out_tok))
}

fn drain_anthropic(events: Vec<AnthropicEvent>) -> Result<(String, String, u32, u32), String> {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut in_tok = 0u32;
    let mut out_tok = 0u32;
    for ev in events {
        match ev {
            AnthropicEvent::Text(s) => text.push_str(&s),
            AnthropicEvent::Reasoning(s) => reasoning.push_str(&s),
            AnthropicEvent::Usage {
                input_tokens,
                output_tokens,
            } => {
                in_tok = input_tokens;
                out_tok = output_tokens;
            }
            AnthropicEvent::Refusal(s) => return Err(format!("refusal: {s}")),
            AnthropicEvent::Done { .. } => {}
        }
    }
    Ok((text, reasoning, in_tok, out_tok))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runa_media::AudioRoutePref;
    use std::path::Path;

    #[test]
    fn prepare_passthrough_without_audio() {
        let cloud = parse_cloud_ref("openai:gpt-4o-mini").unwrap();
        let (p, a) = prepare_cloud_audio(&cloud, "hi", None, AudioRoutePref::Auto).unwrap();
        assert_eq!(p, "hi");
        assert!(a.is_empty());
    }

    #[test]
    fn prepare_native_anthropic_errors() {
        let cloud = parse_cloud_ref("anthropic:claude-sonnet-4").unwrap();
        let err = prepare_cloud_audio(
            &cloud,
            "hi",
            Some(Path::new("/nope.wav")),
            AudioRoutePref::Native,
        )
        .unwrap_err();
        assert!(err.to_lowercase().contains("anthropic"), "{err}");
    }

    #[test]
    fn prepare_native_openai_text_model_errors() {
        let cloud = parse_cloud_ref("openai:gpt-4o-mini").unwrap();
        let err = prepare_cloud_audio(
            &cloud,
            "hi",
            Some(Path::new("/nope.wav")),
            AudioRoutePref::Native,
        )
        .unwrap_err();
        assert!(err.contains("input_audio"), "{err}");
    }
}
