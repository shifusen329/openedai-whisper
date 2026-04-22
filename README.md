OpenedAI Whisper
----------------

Notice: This software is mostly obsolete and will no longer be updated.

Some Alternative(s):

* https://speaches.ai/
* https://github.com/gpustack/vox-box

----

An OpenAI API compatible speech to text server for audio transcription and translations, aka. Whisper.

- Compatible with the OpenAI audio/transcriptions and audio/translations API
- Does not connect to the OpenAI API and does not require an OpenAI API Key
- Not affiliated with OpenAI in any way

API Compatibility:
- [X] /v1/audio/transcriptions
- [X] /v1/audio/translations

Parameter Support:
- [X] `file`
- [X] `model` (all whisper / distil-whisper sizes; see list below)
- [X] `language`
- [X] `prompt` (passed as `initial_prompt` to faster-whisper)
- [X] `temperature`
- [X] `response_format`:
- - [X] `json`
- - [X] `text`
- - [X] `srt`
- - [X] `vtt`
- - [X] `verbose_json`

Details:
* Backend: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2). No PyTorch dependency.
* CUDA or CPU support (automatically detected).
* Compute type auto-selected from `bfloat16` / `float16` / `int8` / `float32` based on device capabilities; override with `-t`.
* **Silero VAD is enabled by default** to suppress hallucinations during silence and non-speech. Set `WHISPER_VAD=0` to disable.
* `condition_on_previous_text` defaults to **off** to avoid repetition-loop hallucinations. Set `WHISPER_CONDITION_PREV=1` to restore faster-whisper's upstream default.

Supported models (public names preserved from the previous HF-transformers backend; mapped internally to faster-whisper ids):
* `openai/whisper-{tiny,base,small,medium,large,large-v2,large-v3,large-v3-turbo}` and their `.en` variants
* `distil-whisper/distil-{small.en,medium.en,large-v2,large-v3}`
* Raw faster-whisper ids (`large-v3`, `distil-large-v3`, ...), `Systran/faster-whisper-*` HF repos, and local CT2 model paths are also accepted via `-m`.

Default model: `openai/whisper-large-v2`.

Version: 0.2.0, Last update: 2026-04-21


API Documentation
-----------------

## Usage

* [OpenAI Speech to text guide](https://platform.openai.com/docs/guides/speech-to-text)
* [OpenAI API Transcription Reference](https://platform.openai.com/docs/api-reference/audio/createTranscription)
* [OpenAI API Translation Reference](https://platform.openai.com/docs/api-reference/audio/createTranslation)


Installation instructions
-------------------------

You will need a recent NVIDIA driver (>= 525, supporting CUDA 12) for GPU inference. Faster-whisper wheels on PyPI bundle the required CTranslate2 CUDA libraries; you don't need a full CUDA toolkit install.

```shell
# Install the Python requirements
pip install -r requirements.txt
# ffmpeg is optional (PyAV bundles decoders) but harmless to install
sudo apt install ffmpeg
```

Usage
-----

```
Usage: whisper.py [-m <model_name>] [-d <device>] [-t <dtype>] [-P <port>] [-H <host>] [--preload]


Description:
OpenedAI Whisper API Server

Options:
-h, --help            Show this help message and exit.
-m MODEL, --model MODEL
                      The model to use for transcription.
                      Ex. distil-whisper/distil-medium.en (default: openai/whisper-large-v2)
-d DEVICE, --device DEVICE
                      Device for inference: auto, cuda, or cpu (default: auto)
--device-index INDEX  CUDA device index when device=cuda (default: 0)
-t DTYPE, --dtype DTYPE
                      Compute type: auto, float32, float16, bfloat16, int8, int8_float16 (default: auto)
-P PORT, --port PORT  Server tcp port (default: 8000)
-H HOST, --host HOST  Host to listen on, Ex. 0.0.0.0 (default: localhost)
--preload             Preload model and exit. (default: False)
```

Sample API Usage
----------------

You can use it like this:

```shell
curl -s http://localhost:8000/v1/audio/transcriptions -H "Content-Type: multipart/form-data" -F model="whisper-1" -F file="@audio.mp3" -F response_format=text
```

Or just like this:

```shell
curl -s http://localhost:8000/v1/audio/transcriptions -F model="whisper-1" -F file="@audio.mp3"
```

Or like this example from the [OpenAI Speech to text guide Quickstart](https://platform.openai.com/docs/guides/speech-to-text/quickstart):

```python
from openai import OpenAI
client = OpenAI(api_key='sk-1111', base_url='http://localhost:8000/v1')

audio_file = open("/path/to/file/audio.mp3", "rb")
transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
print(transcription.text)
```

Docker support
--------------

You can run the server via docker like so:
```shell
docker compose build
docker compose up
```

Options can be set via `whisper.env`.
