import tempfile
import time

import scipy.io.wavfile as wav
from scipy import signal
import sounddevice as sd
import soundfile as sf
import streamlit as st
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    VitsModel,
    pipeline,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIPELINE_DEVICE = 0 if torch.cuda.is_available() else -1


def record_audio(duration=5, fs=16000):
    progress = st.progress(0, text="Recording...")
    status = st.empty()
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)

    start_time = time.time()
    while sd.get_stream().active:
        elapsed = min(time.time() - start_time, duration)
        percent = int((elapsed / duration) * 100)
        progress.progress(percent, text=f"Recording... {elapsed:.1f}s / {duration}s")
        status.caption("|" * max(1, percent // 5))
        time.sleep(0.1)

    sd.wait()
    progress.progress(100, text="Recording complete")
    status.caption("|" * 20)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp.name, fs, audio)
    return temp.name


@st.cache_resource
def load_asr():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        framework="pt",
        device=PIPELINE_DEVICE,
    )


@st.cache_resource
def load_llm():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(DEVICE)
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=PIPELINE_DEVICE,
    )


@st.cache_resource
def load_tts():
    model_name = "facebook/mms-tts-eng"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name).to(DEVICE)
    return tokenizer, model


def transcribe(asr, audio_path):
    audio, sample_rate = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    target_rate = asr.feature_extractor.sampling_rate
    if sample_rate != target_rate:
        audio = signal.resample_poly(audio, target_rate, sample_rate)

    audio = audio.astype("float32")
    return asr({"raw": audio, "sampling_rate": target_rate})["text"]


def generate_response(llm, text):
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return "I could not hear enough clear speech to respond."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant for field notes. "
                "Give a clear, natural, concise response. "
                "If the note mentions a problem, include a practical next step."
            ),
        },
        {
            "role": "user",
            "content": cleaned,
        },
    ]
    output = llm(
        messages,
        max_new_tokens=80,
        do_sample=False,
        truncation=True,
        return_full_text=False,
    )
    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        return generated[-1]["content"].strip()
    return str(generated).strip()


def text_to_speech(tokenizer, model, text):
    inputs = tokenizer(text=text, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.inference_mode():
        speech = model(**inputs).waveform

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(
        output_file.name,
        speech.squeeze(0).detach().cpu().numpy(),
        samplerate=model.config.sampling_rate,
    )
    return output_file.name


def show_model_load_error(stage, error):
    st.error(
        f"{stage} model could not be loaded.\n\n"
        "This app downloads models from Hugging Face the first time it runs. "
        "Your current network or proxy settings are blocking that download.\n\n"
        f"Details: {error}"
    )
    st.info(
        "Fix options:\n"
        "1. Connect to the internet and allow access to huggingface.co.\n"
        "2. Correct or remove broken HTTP/HTTPS proxy environment variables.\n"
        "3. Pre-download the models, then run the app again."
    )
    st.stop()


st.title("Voice AI Assistant")
st.write("Record or upload audio -> Transcribe -> Generate response -> Hear response")

option = st.radio("Choose Input Method", ["Record Audio", "Upload File"])
audio_path = None

if option == "Record Audio":
    if st.button("Record 5 seconds"):
        audio_path = record_audio()
        st.success("Recording complete!")
elif option == "Upload File":
    uploaded = st.file_uploader("Upload WAV file", type=["wav"])
    if uploaded:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp.write(uploaded.read())
        audio_path = temp.name

if audio_path:
    st.audio(audio_path)

    try:
        asr = load_asr()
    except Exception as error:
        show_model_load_error("Speech recognition", error)

    with st.spinner("Transcribing..."):
        text = transcribe(asr, audio_path)

    st.subheader("Transcription")
    st.write(text)

    try:
        llm = load_llm()
    except Exception as error:
        show_model_load_error("Response model", error)

    with st.spinner("Generating response..."):
        response = generate_response(llm, text)

    st.subheader("Assistant Response")
    st.write(response)

    try:
        tts_tokenizer, tts_model = load_tts()
    except Exception as error:
        show_model_load_error("Text-to-speech", error)

    with st.spinner("Converting to speech..."):
        output_audio = text_to_speech(tts_tokenizer, tts_model, response)

    st.subheader("Audio Output")
    st.audio(output_audio)
