# Voice AI Assistant

A local Streamlit-based voice assistant that:

- records audio or accepts a WAV upload
- transcribes speech with Whisper
- generates a text response with Qwen
- converts the response back to speech with MMS TTS

The app is implemented in [voice.py](/c:/Users/ASH-EEM/Desktop/AICN/Assignment/voice.py).

## Features

- `Record Audio` mode for 5-second microphone capture
- `Upload File` mode for WAV files
- speech-to-text using `openai/whisper-base`
- response generation using `Qwen/Qwen2.5-3B-Instruct`
- text-to-speech using `facebook/mms-tts-eng`
- automatic GPU usage when CUDA is available
- first-run model caching with Streamlit resource cache

## Models Used

- ASR: `openai/whisper-base`
- LLM: `Qwen/Qwen2.5-3B-Instruct`
- TTS: `facebook/mms-tts-eng`

## Requirements

Install these Python packages:

```bash
pip install streamlit torch transformers sounddevice soundfile scipy
```

Recommended:

- Python 3.10+
- NVIDIA GPU with CUDA-enabled PyTorch for faster inference
- internet access on first run to download Hugging Face models

## How It Works

1. The user records audio or uploads a WAV file.
2. The app transcribes the audio with Whisper.
3. The transcript is sent to Qwen with a short assistant-style prompt.
4. The generated response is converted to speech.
5. The response text and generated audio are shown in the UI.

## Run the App

From the project directory, run:

```bash
streamlit run Assignment/voice.py
```

Or from inside the `Assignment` folder:

```bash
streamlit run voice.py
```

## GPU Support

The app automatically checks for CUDA:

- if CUDA is available, it uses GPU
- otherwise it falls back to CPU

GPU is especially helpful for:

- Whisper transcription
- Qwen response generation
- text-to-speech inference

## Notes About First Run

The first run may take several minutes because the models must be downloaded and cached locally.

After the models are cached:

- startup is faster
- inference is faster
- the app can run locally without an API key

## Known Limitations

- recording is fixed at 5 seconds in the current UI
- uploaded files must be WAV
- large language model responses can still be slow compared to smaller models
- first-time model downloads require access to `huggingface.co`
