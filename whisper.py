#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio
import json
import logging
import time
import threading
import gc
import io

import numpy as np
import ctranslate2
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import get_speech_timestamps, VadOptions
from typing import Optional, List
from fastapi import UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

import openedai

model = None
last_usage_time = None
model_config = None
unload_timer = None
is_english_only = False
current_model_name = None
app = openedai.OpenAIStub()

# Set to False to preload the model at startup instead of on first request
LAZY_LOAD = True

# Seconds of inactivity before unloading the model (0 to disable unloading)
UNLOAD_TIMEOUT = 300

# Silero VAD is on by default to suppress silence hallucinations. Set WHISPER_VAD=0 to disable.
VAD_FILTER_DEFAULT = os.environ.get("WHISPER_VAD", "1") != "0"
# Disable condition_on_previous_text by default to prevent repetition-loop hallucinations.
# Set WHISPER_CONDITION_PREV=1 to restore faster-whisper's upstream default.
CONDITION_PREV_DEFAULT = os.environ.get("WHISPER_CONDITION_PREV", "0") == "1"

_transcribe_lock = threading.Lock()

# Speaker diarization (pyannote.audio) — lazy-loaded on first diarize=true request
_diarize_pipeline = None
_diarize_load_lock = threading.Lock()

# WebSocket streaming defaults (Deepgram-shape /v1/listen)
WS_SAMPLE_RATE = 16000
WS_VAD_MIN_SILENCE_MS = 500     # silence gap that closes a speech segment
WS_VAD_CHECK_EVERY_MS = 250     # how often to re-evaluate the buffer
WS_MAX_BUFFER_SECONDS = 30      # safety cap on retained audio
WS_MIN_SEGMENT_MS = 200         # ignore VAD blips shorter than this

# Available whisper models (public names; mapped internally to faster-whisper ids)
AVAILABLE_MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-tiny.en",
    "openai/whisper-base",
    "openai/whisper-base.en",
    "openai/whisper-small",
    "openai/whisper-small.en",
    "openai/whisper-medium",
    "openai/whisper-medium.en",
    "openai/whisper-large",
    "openai/whisper-large-v2",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "distil-whisper/distil-small.en",
    "distil-whisper/distil-medium.en",
    "distil-whisper/distil-large-v2",
    "distil-whisper/distil-large-v3",
]

# Map public HF-style names to faster-whisper model identifiers
FW_MODEL_MAP = {
    "openai/whisper-tiny": "tiny",
    "openai/whisper-tiny.en": "tiny.en",
    "openai/whisper-base": "base",
    "openai/whisper-base.en": "base.en",
    "openai/whisper-small": "small",
    "openai/whisper-small.en": "small.en",
    "openai/whisper-medium": "medium",
    "openai/whisper-medium.en": "medium.en",
    "openai/whisper-large": "large-v1",
    "openai/whisper-large-v2": "large-v2",
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
    "distil-whisper/distil-small.en": "distil-small.en",
    "distil-whisper/distil-medium.en": "distil-medium.en",
    "distil-whisper/distil-large-v2": "distil-large-v2",
    "distil-whisper/distil-large-v3": "distil-large-v3",
}

# Distil models are English-only but lack the .en suffix
EN_ONLY_EXTRA = {"distil-large-v2", "distil-large-v3"}

# Available TTS models (served by chatterbox-api or voxcpm)
TTS_MODELS = [
    "chatterbox",
    "chatterbox-turbo",
    "chatterbox-ririka",
    "voxcpm-tts",
]

default_model = None

def unload_model():
    global model, last_usage_time, unload_timer, current_model_name
    if model is not None and last_usage_time is not None:
        if time.time() - last_usage_time >= UNLOAD_TIMEOUT:
            logging.info("Unloading model due to inactivity")
            model = None
            gc.collect()
            last_usage_time = None
            unload_timer = None
            current_model_name = None
            return

    # Schedule next check
    unload_timer = threading.Timer(30.0, unload_model)
    unload_timer.daemon = True
    unload_timer.start()

def resolve_model_name(model_name: str) -> str:
    """Resolve short model names to the project's public HF-style name."""
    if not model_name or model_name == "whisper-1":
        return default_model
    if "/" in model_name:
        return model_name
    full_name = f"openai/{model_name}"
    if full_name in AVAILABLE_MODELS:
        return full_name
    full_name = f"distil-whisper/{model_name}"
    if full_name in AVAILABLE_MODELS:
        return full_name
    return model_name

def to_fw_id(public_name: str) -> str:
    """Translate a public HF-style name to a faster-whisper model id."""
    return FW_MODEL_MAP.get(public_name, public_name)

def is_english_only_fw(fw_id: str) -> bool:
    return fw_id.endswith(".en") or fw_id in EN_ONLY_EXTRA

def ensure_model_loaded(requested_model: str = None):
    global model, last_usage_time, unload_timer, model_config, is_english_only, current_model_name, default_model

    public_name = resolve_model_name(requested_model)

    if model is not None and current_model_name != public_name:
        logging.info(f"Switching model from {current_model_name} to {public_name}")
        model = None
        gc.collect()

    if model is None:
        fw_id = to_fw_id(public_name)
        logging.info(f"Loading model: {public_name} (faster-whisper id: {fw_id})")
        device, compute_type, device_index, _ = model_config
        is_english_only = is_english_only_fw(fw_id)
        model = WhisperModel(fw_id, device=device, device_index=device_index, compute_type=compute_type)
        current_model_name = public_name

    last_usage_time = time.time()

    if UNLOAD_TIMEOUT > 0:
        if unload_timer is not None:
            unload_timer.cancel()
        unload_timer = threading.Timer(30.0, unload_model)
        unload_timer.daemon = True
        unload_timer.start()


def _run_transcribe(audio_bytes: bytes, word_timestamps: bool, fw_kwargs: dict) -> dict:
    """Run transcription and return a dict compatible with the legacy response builders.

    Decodes audio once into a 16kHz mono float32 array so the same buffer can be reused
    by diarization without a second PyAV decode pass.
    """
    audio_f32 = decode_audio(io.BytesIO(audio_bytes), sampling_rate=16000)
    with _transcribe_lock:
        seg_iter, info = model.transcribe(
            audio_f32,
            word_timestamps=word_timestamps,
            **fw_kwargs,
        )
        # segments is a generator; materializing drives inference and populates info.duration
        segments = list(seg_iter)

    if word_timestamps:
        chunks = [
            {"text": w.word, "timestamp": (w.start, w.end)}
            for s in segments for w in (s.words or [])
        ]
    else:
        chunks = [
            {"text": s.text, "timestamp": (s.start, s.end)}
            for s in segments
        ]
    text = "".join(s.text for s in segments)
    return {"text": text, "chunks": chunks, "info": info, "audio_f32": audio_f32}


def _run_transcribe_array(audio_f32: np.ndarray, fw_kwargs: dict) -> dict:
    """Like _run_transcribe but accepts a numpy float32 array directly (skips PyAV decode)."""
    with _transcribe_lock:
        seg_iter, info = model.transcribe(audio_f32, **fw_kwargs)
        segments = list(seg_iter)

    chunks = []
    if fw_kwargs.get("word_timestamps"):
        for s in segments:
            for w in (s.words or []):
                chunks.append({"text": w.word, "start": w.start, "end": w.end})
    else:
        for s in segments:
            chunks.append({"text": s.text, "start": s.start, "end": s.end})
    text = "".join(s.text for s in segments)
    return {"text": text, "chunks": chunks, "info": info}


def ensure_diarizer_loaded():
    """Lazy-load the pyannote speaker diarization pipeline onto the same GPU as whisper."""
    global _diarize_pipeline
    if _diarize_pipeline is not None:
        return
    with _diarize_load_lock:
        if _diarize_pipeline is not None:
            return
        from pyannote.audio import Pipeline
        import torch
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        logging.info("Loading pyannote/speaker-diarization-3.1")
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        if pipe is None:
            raise RuntimeError(
                "pyannote pipeline returned None — set HF_TOKEN and accept EULAs for "
                "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0"
            )
        device_index = model_config[2] if model_config else 0
        device = "cuda" if (model_config and model_config[0] == "cuda") else "cpu"
        target = torch.device(f"{device}:{device_index}" if device == "cuda" else "cpu")
        pipe.to(target)
        _diarize_pipeline = pipe
        logging.info(f"pyannote pipeline ready on {target}")


def _run_diarize(audio_f32: np.ndarray, min_speakers=None, max_speakers=None) -> list:
    """Run speaker diarization on a 16kHz mono float32 array.

    Returns a list of {"start", "end", "speaker"} segments where speaker is an int
    assigned in first-seen order (SPEAKER_00 -> 0, SPEAKER_01 -> 1, ...).
    """
    ensure_diarizer_loaded()
    import torch
    waveform = torch.from_numpy(audio_f32).unsqueeze(0)  # (1, samples)
    with _transcribe_lock:                                # share GPU lock with whisper
        diarization = _diarize_pipeline(
            {"waveform": waveform, "sample_rate": 16000},
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    label_to_int = {}
    out = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in label_to_int:
            label_to_int[speaker] = len(label_to_int)
        out.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": label_to_int[speaker],
        })
    return out


def _assign_speakers_to_chunks(chunks: list, diar_segments: list) -> None:
    """Mutate chunks in place; pick the speaker whose turn maximally overlaps each chunk."""
    for c in chunks:
        c_start, c_end = c["timestamp"]
        best_spk, best_overlap = 0, 0.0
        for ds in diar_segments:
            ov = max(0.0, min(c_end, ds["end"]) - max(c_start, ds["start"]))
            if ov > best_overlap:
                best_overlap, best_spk = ov, ds["speaker"]
        c["speaker"] = best_spk


def _deepgram_event(text, words, start, duration, is_final=True):
    """Build a Deepgram-shape Results event matching what OMI's stt_response_schema parses."""
    return {
        "type": "Results",
        "channel_index": [0, 1],
        "start": start,
        "duration": duration,
        "is_final": is_final,
        "speech_final": is_final,
        "channel": {
            "alternatives": [{
                "transcript": text,
                "confidence": 1.0,
                "words": [
                    {"word": w["text"].strip().lower(),
                     "punctuated_word": w["text"].strip(),
                     "start": w["start"],
                     "end": w["end"],
                     "speaker": 0,
                     "confidence": 1.0}
                    for w in words
                ],
            }]
        },
    }


async def whisper(file, response_format: str, word_timestamps: bool, task: str, fw_kwargs: dict,
                  diarize: bool = False, min_speakers=None, max_speakers=None):
    if diarize and response_format != "verbose_json":
        return JSONResponse(
            status_code=400,
            content={"error": "diarize=true requires response_format=verbose_json"},
        )

    result = _run_transcribe(await file.read(), word_timestamps, fw_kwargs)

    filename_noext, ext = os.path.splitext(file.filename)
    info = result["info"]
    chunks = result["chunks"]

    if diarize:
        try:
            diar = _run_diarize(result["audio_f32"], min_speakers, max_speakers)
        except Exception as e:
            logging.exception("diarization failed")
            return JSONResponse(
                status_code=503,
                content={"error": f"diarization failed: {e}"},
            )
        _assign_speakers_to_chunks(chunks, diar)

    if response_format == "text":
        return PlainTextResponse(result["text"].strip(), headers={"Content-Disposition": f"attachment; filename={filename_noext}.txt"})

    elif response_format == "json":
        return JSONResponse(content={ 'text': result['text'].strip() }, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})

    elif response_format == "verbose_json":
        duration = info.duration if info.duration else (chunks[-1]['timestamp'][1] if chunks else 0.0)
        response = {
            "task": task,
            "language": info.language,
            "duration": duration,
            "text": result["text"].strip(),
        }
        if word_timestamps:
            response['words'] = [
                {
                    'word': c['text'].strip(),
                    'start': c['timestamp'][0],
                    'end': c['timestamp'][1],
                    **({'speaker': c['speaker']} if 'speaker' in c else {}),
                } for c in chunks
            ]
        else:
            response['segments'] = [
                {
                    'id': i,
                    'start': c['timestamp'][0],
                    'end': c['timestamp'][1],
                    'text': c['text'].strip(),
                    **({'speaker': c['speaker']} if 'speaker' in c else {}),
                } for i, c in enumerate(chunks)
            ]

        return JSONResponse(content=response, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}_verbose.json"})

    elif response_format == "srt":
            def srt_time(t):
                return "{:02d}:{:02d}:{:06.3f}".format(int(t//3600), int(t//60)%60, t%60).replace(".", ",")

            return PlainTextResponse("\n".join([ f"{i}\n{srt_time(c['timestamp'][0])} --> {srt_time(c['timestamp'][1])}\n{c['text'].strip()}\n"
                for i, c in enumerate(chunks, 1) ]), media_type="text/srt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.srt"})

    elif response_format == "vtt":
            def vtt_time(t):
                return "{:02d}:{:06.3f}".format(int(t//60), t%60)

            return PlainTextResponse("\n".join(["WEBVTT\n"] + [ f"{vtt_time(c['timestamp'][0])} --> {vtt_time(c['timestamp'][1])}\n{c['text'].strip()}\n"
                for c in chunks ]), media_type="text/vtt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.vtt"})


def _build_fw_kwargs(task, language, prompt, temperature, response_format,
                     timestamp_granularities, english_only):
    k = {
        "task": task,
        "vad_filter": VAD_FILTER_DEFAULT,
        "vad_parameters": {"min_silence_duration_ms": 500},
        "condition_on_previous_text": CONDITION_PREV_DEFAULT,
        "initial_prompt": prompt or None,
    }
    if english_only:
        k["language"] = "en"
    elif language:
        k["language"] = language
    if temperature is not None:
        k["temperature"] = temperature
    return k


@app.get("/v1/audio/transcriptions/openapi.json", include_in_schema=False)
async def transcriptions_openapi():
    """Mirror of /openapi.json reachable through the FQDN, since nginx routes
    only the /v1/audio/transcriptions prefix to this server."""
    return app.openapi()


@app.post("/v1/audio/transcriptions")
async def transcriptions(
        file: UploadFile,
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
        timestamp_granularities: List[str] = Form(["segment"]),
        diarize: Optional[bool] = Form(False),
        min_speakers: Optional[int] = Form(None),
        max_speakers: Optional[int] = Form(None),
    ):
    global is_english_only

    ensure_model_loaded(model)

    word_timestamps = response_format == "verbose_json" and "word" in timestamp_granularities
    fw_kwargs = _build_fw_kwargs(
        task="transcribe",
        language=language,
        prompt=prompt,
        temperature=temperature,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
        english_only=is_english_only,
    )
    return await whisper(file, response_format, word_timestamps, "transcribe", fw_kwargs,
                         diarize=diarize, min_speakers=min_speakers, max_speakers=max_speakers)


@app.post("/v1/audio/translations")
async def translations(
        file: UploadFile,
        model: str = Form(...),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
        diarize: Optional[bool] = Form(False),
        min_speakers: Optional[int] = Form(None),
        max_speakers: Optional[int] = Form(None),
    ):
    global is_english_only

    ensure_model_loaded(model)

    if is_english_only:
        return JSONResponse(
            status_code=400,
            content={"error": "Translation is not supported for English-only models"}
        )

    word_timestamps = False
    fw_kwargs = _build_fw_kwargs(
        task="translate",
        language=None,
        prompt=prompt,
        temperature=temperature,
        response_format=response_format,
        timestamp_granularities=["segment"],
        english_only=False,
    )
    return await whisper(file, response_format, word_timestamps, "translate", fw_kwargs,
                         diarize=diarize, min_speakers=min_speakers, max_speakers=max_speakers)


@app.websocket("/v1/listen")
async def listen(websocket: WebSocket):
    """Deepgram-shape streaming STT. Accepts raw PCM16 mono @ 16kHz binary frames."""
    qp = websocket.query_params
    encoding = qp.get("encoding", "linear16")
    try:
        sample_rate = int(qp.get("sample_rate", str(WS_SAMPLE_RATE)))
    except ValueError:
        sample_rate = -1
    requested_model = qp.get("model")
    language = qp.get("language")
    initial_prompt = qp.get("prompt")

    await websocket.accept()

    if encoding != "linear16" or sample_rate != WS_SAMPLE_RATE:
        await websocket.send_json({"type": "Error", "message":
            f"only encoding=linear16, sample_rate={WS_SAMPLE_RATE} supported (got {encoding}, {sample_rate})"})
        await websocket.close(code=1003)
        return

    try:
        ensure_model_loaded(requested_model)
    except Exception as e:
        await websocket.send_json({"type": "Error", "message": f"model load failed: {e}"})
        await websocket.close(code=1011)
        return

    english_only = is_english_only
    pcm_buf = bytearray()
    consumed_samples = 0
    last_vad_check_samples = 0
    bytes_per_sample = 2

    vad_opts = VadOptions(min_silence_duration_ms=WS_VAD_MIN_SILENCE_MS,
                          min_speech_duration_ms=WS_MIN_SEGMENT_MS)

    async def _flush(force_final: bool):
        nonlocal pcm_buf, consumed_samples, last_vad_check_samples
        if len(pcm_buf) < int(bytes_per_sample * sample_rate * 0.2):
            return
        audio = (np.frombuffer(bytes(pcm_buf), dtype=np.int16)
                   .astype(np.float32) / 32768.0)

        speech = get_speech_timestamps(audio, vad_opts, sampling_rate=sample_rate)
        if not speech:
            keep = int(0.5 * sample_rate) * bytes_per_sample
            if len(pcm_buf) > keep:
                consumed_samples += (len(pcm_buf) - keep) // bytes_per_sample
                del pcm_buf[:-keep]
            return

        last = speech[-1]
        tail_silence_samples = len(audio) - last["end"]
        min_silence_samples = int(WS_VAD_MIN_SILENCE_MS / 1000 * sample_rate)
        if not force_final and tail_silence_samples < min_silence_samples:
            return  # current speech still ongoing

        seg_start = speech[0]["start"]
        seg_end = last["end"]
        segment = audio[seg_start:seg_end]

        fw_kwargs = {
            "task": "transcribe",
            "vad_filter": False,
            "condition_on_previous_text": CONDITION_PREV_DEFAULT,
            "initial_prompt": initial_prompt,
            "word_timestamps": True,
        }
        if english_only:
            fw_kwargs["language"] = "en"
        elif language:
            fw_kwargs["language"] = language

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _run_transcribe_array(segment, fw_kwargs))

        seg_offset_s = (consumed_samples + seg_start) / sample_rate
        words = [
            {"text": w["text"], "start": seg_offset_s + w["start"], "end": seg_offset_s + w["end"]}
            for w in result["chunks"]
        ]
        text = result["text"].strip()
        if text:
            await websocket.send_json(_deepgram_event(
                text, words,
                start=seg_offset_s,
                duration=(seg_end - seg_start) / sample_rate,
                is_final=True,
            ))

        bytes_consumed = seg_end * bytes_per_sample
        consumed_samples += seg_end
        del pcm_buf[:bytes_consumed]
        last_vad_check_samples = 0

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                pcm_buf.extend(msg["bytes"])
                max_bytes = WS_MAX_BUFFER_SECONDS * sample_rate * bytes_per_sample
                if len(pcm_buf) > max_bytes:
                    drop = len(pcm_buf) - max_bytes
                    consumed_samples += drop // bytes_per_sample
                    del pcm_buf[:drop]
                samples_now = len(pcm_buf) // bytes_per_sample
                check_every_samples = int(WS_VAD_CHECK_EVERY_MS / 1000 * sample_rate)
                if samples_now - last_vad_check_samples >= check_every_samples:
                    last_vad_check_samples = samples_now
                    await _flush(force_final=False)
            elif msg.get("text") is not None:
                try:
                    if json.loads(msg["text"]).get("type") == "CloseStream":
                        break
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass

    try:
        await _flush(force_final=True)
        await websocket.send_json({"type": "Metadata", "request_id": "n/a"})
    except Exception:
        pass
    try:
        await websocket.close()
    except Exception:
        pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='whisper.py',
        description='OpenedAI Whisper API Server (faster-whisper backend)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model', action='store', default="openai/whisper-large-v2", help="The model to use. Accepts HF-style names (openai/whisper-large-v2), short names (whisper-large-v2), raw faster-whisper ids (large-v3, distil-large-v3), CT2 HF repos (Systran/faster-whisper-large-v3), or a local path.")
    parser.add_argument('-d', '--device', action='store', default="auto", help="Device for inference: auto, cuda, or cpu.")
    parser.add_argument('-t', '--dtype', action='store', default="auto", help="Compute type: auto, float32, float16, bfloat16, int8, int8_float16.")
    parser.add_argument('--device-index', action='store', default=0, type=int, help="CUDA device index when device=cuda.")
    parser.add_argument('-P', '--port', action='store', default=8000, type=int, help="Server tcp port")
    parser.add_argument('-H', '--host', action='store', default='localhost', help="Host to listen on, Ex. 0.0.0.0")
    parser.add_argument('--preload', action='store_true', help="Preload model and exit.")

    return parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _pick_compute_type(device: str, requested: str) -> str:
    supported = set(ctranslate2.get_supported_compute_types(device))
    if requested == "auto":
        for candidate in ("bfloat16", "float16"):
            if candidate in supported:
                return candidate
        if device == "cpu" and "int8" in supported:
            return "int8"
        return "float32"
    if requested in supported:
        return requested
    fallback = "float16" if "float16" in supported else "float32"
    print(f"compute type '{requested}' not supported on {device}; falling back to '{fallback}'", file=sys.stderr)
    return fallback

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    else:
        device = args.device

    compute_type = _pick_compute_type(device, args.dtype)

    model_config = (device, compute_type, args.device_index, args.model)
    default_model = args.model

    if args.preload:
        fw_id = to_fw_id(resolve_model_name(args.model))
        logging.info(f"Preloading model: {args.model} (faster-whisper id: {fw_id}) on {device} [{compute_type}]")
        _preload = WhisperModel(fw_id, device=device, device_index=args.device_index, compute_type=compute_type)
        sys.exit(0)

    app.register_model('whisper-1', args.model, model_type='stt')

    # Register all available STT models
    for model_id in AVAILABLE_MODELS:
        app.register_model(model_id, model_type='stt')
        if model_id.startswith('openai/'):
            short_name = model_id.replace('openai/', '', 1)
            app.register_model(short_name, model_id, model_type='stt')

    # Register TTS models
    for model_id in TTS_MODELS:
        app.register_model(model_id, model_type='tts')

    # Preload model at startup if lazy loading is disabled
    if not LAZY_LOAD:
        logging.info(f"Preloading model: {default_model}")
        ensure_model_loaded()

    uvicorn.run(app, host=args.host, port=args.port) # , root_path=cwd, access_log=False, log_level="info", ssl_keyfile="cert.pem", ssl_certfile="cert.pem")
