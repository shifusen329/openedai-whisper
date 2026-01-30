#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time
import threading

import torch
from transformers import pipeline
from typing import Optional, List
from fastapi import UploadFile, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

import openedai

pipe = None
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

# Available whisper models
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

# Available TTS models (served by chatterbox-api or voxcpm)
TTS_MODELS = [
    "chatterbox",
    "chatterbox-turbo",
    "chatterbox-ririka",
    "voxcpm-tts",
]

default_model = None

def unload_model():
    global pipe, last_usage_time, unload_timer, current_model_name
    if pipe is not None and last_usage_time is not None:
        if time.time() - last_usage_time >= UNLOAD_TIMEOUT:
            logging.info("Unloading model due to inactivity")
            pipe.model = pipe.model.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipe = None
            last_usage_time = None
            unload_timer = None
            current_model_name = None
            return
    
    # Schedule next check
    unload_timer = threading.Timer(30.0, unload_model)
    unload_timer.daemon = True
    unload_timer.start()

def ensure_model_loaded(requested_model: str = None):
    global pipe, last_usage_time, unload_timer, model_config, is_english_only, current_model_name, default_model

    # Use requested model, or fall back to default
    model_name = requested_model if requested_model and requested_model != "whisper-1" else default_model

    # Check if we need to load a different model
    if pipe is not None and current_model_name != model_name:
        logging.info(f"Switching model from {current_model_name} to {model_name}")
        pipe.model = pipe.model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pipe = None

    if pipe is None:
        logging.info(f"Loading model: {model_name}")
        device, dtype, _ = model_config
        is_english_only = model_name.endswith('.en')
        pipe = pipeline("automatic-speech-recognition", model=model_name, device=device, chunk_length_s=30, torch_dtype=dtype)
        current_model_name = model_name

    last_usage_time = time.time()

    # Start or restart the unload timer (skip if unloading is disabled)
    if UNLOAD_TIMEOUT > 0:
        if unload_timer is not None:
            unload_timer.cancel()
        unload_timer = threading.Timer(30.0, unload_model)
        unload_timer.daemon = True
        unload_timer.start()

async def whisper(file, response_format: str, **kwargs):
    global pipe
    
    ensure_model_loaded()
    result = pipe(await file.read(), **kwargs)

    filename_noext, ext = os.path.splitext(file.filename)

    if response_format == "text":
        return PlainTextResponse(result["text"].strip(), headers={"Content-Disposition": f"attachment; filename={filename_noext}.txt"})

    elif response_format == "json":
        return JSONResponse(content={ 'text': result['text'].strip() }, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})
    
    elif response_format == "verbose_json":
        chunks = result["chunks"]

        response = {
            "task": kwargs['generate_kwargs'].get('task', 'transcribe'),
            #"language": "english",
            "duration": chunks[-1]['timestamp'][1],
            "text": result["text"].strip(),
        }
        if kwargs['return_timestamps'] == 'word':
            response['words'] = [{'word': chunk['text'].strip(), 'start': chunk['timestamp'][0], 'end': chunk['timestamp'][1] } for chunk in chunks ]
        else:
            response['segments'] = [{
                    "id": i,
                    #"seek": 0,
                    'start': chunk['timestamp'][0],
                    'end': chunk['timestamp'][1],
                    'text': chunk['text'].strip(),
                    #"tokens": [ ],
                    #"temperature": 0.0,
                    #"avg_logprob": -0.2860786020755768,
                    #"compression_ratio": 1.2363636493682861,
                    #"no_speech_prob": 0.00985979475080967
            } for i, chunk in enumerate(chunks) ]
        
        return JSONResponse(content=response, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}_verbose.json"})

    elif response_format == "srt":
            def srt_time(t):
                return "{:02d}:{:02d}:{:06.3f}".format(int(t//3600), int(t//60)%60, t%60).replace(".", ",")

            return PlainTextResponse("\n".join([ f"{i}\n{srt_time(chunk['timestamp'][0])} --> {srt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                for i, chunk in enumerate(result["chunks"], 1) ]), media_type="text/srt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.srt"})

    elif response_format == "vtt":
            def vtt_time(t):
                return "{:02d}:{:06.3f}".format(int(t//60), t%60)
            
            return PlainTextResponse("\n".join(["WEBVTT\n"] + [ f"{vtt_time(chunk['timestamp'][0])} --> {vtt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                for chunk in result["chunks"] ]), media_type="text/vtt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.vtt"})


@app.post("/v1/audio/transcriptions")
async def transcriptions(
        file: UploadFile,
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
        timestamp_granularities: List[str] = Form(["segment"])
    ):
    global pipe, is_english_only

    ensure_model_loaded(model)

    kwargs = {'generate_kwargs': {}}

    # English-only models don't support task or language parameters
    if not is_english_only:
        kwargs['generate_kwargs']['task'] = 'transcribe'
        if language:
            kwargs['generate_kwargs']["language"] = language
# May work soon, https://github.com/huggingface/transformers/issues/27317
#    if prompt:
#        kwargs["initial_prompt"] = prompt
    if temperature:
        kwargs['generate_kwargs']["temperature"] = temperature
        kwargs['generate_kwargs']['do_sample'] = True

    if response_format == "verbose_json" and 'word' in timestamp_granularities:
        kwargs['return_timestamps'] = 'word'
    else:
        kwargs['return_timestamps'] = response_format in ["verbose_json", "srt", "vtt"]

    return await whisper(file, response_format, **kwargs)


@app.post("/v1/audio/translations")
async def translations(
        file: UploadFile,
        model: str = Form(...),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
    ):
    global pipe, is_english_only

    ensure_model_loaded(model)

    # English-only models don't support translation (only transcribe)
    if is_english_only:
        return JSONResponse(
            status_code=400,
            content={"error": "Translation is not supported for English-only models"}
        )

    kwargs = {'generate_kwargs': {"task": "translate"}}

# May work soon, https://github.com/huggingface/transformers/issues/27317
#    if prompt:
#        kwargs["initial_prompt"] = prompt
    if temperature:
        kwargs['generate_kwargs']["temperature"] = temperature
        kwargs['generate_kwargs']['do_sample'] = True

    kwargs['return_timestamps'] = response_format in ["verbose_json", "srt", "vtt"]

    return await whisper(file, response_format, **kwargs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='whisper.py',
        description='OpenedAI Whisper API Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model', action='store', default="openai/whisper-large-v2", help="The model to use for transcription. Ex. distil-whisper/distil-medium.en")
    parser.add_argument('-d', '--device', action='store', default="auto", help="Set the torch device for the model. Ex. cuda:1")
    parser.add_argument('-t', '--dtype', action='store', default="auto", help="Set the torch data type for processing (float32, float16, bfloat16)")
    parser.add_argument('-P', '--port', action='store', default=8000, type=int, help="Server tcp port")
    parser.add_argument('-H', '--host', action='store', default='localhost', help="Host to listen on, Ex. 0.0.0.0")
    parser.add_argument('--preload', action='store_true', help="Preload model and exit.")

    return parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dtype == "auto":
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 if args.dtype == "float16" else torch.float32

        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("bfloat16 not supported on this hardware, falling back to float16", file=sys.stderr)
            dtype = torch.float16

    model_config = (device, dtype, args.model)
    default_model = args.model

    if args.preload:
        pipe = pipeline("automatic-speech-recognition", model=args.model, device=device, chunk_length_s=30, torch_dtype=dtype)
        sys.exit(0)

    app.register_model('whisper-1', args.model, model_type='stt')

    # Register all available STT models
    for model_id in AVAILABLE_MODELS:
        app.register_model(model_id, model_type='stt')
        # Also register without the openai/ prefix as an alias
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
