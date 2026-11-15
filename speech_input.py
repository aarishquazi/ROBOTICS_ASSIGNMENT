"""Optional local microphone input using OpenAI Whisper.

Whisper automatically detects the spoken language when language=None. The
multilingual model then translates non-English speech into English with
task='translate'. Audio is recorded locally and processed locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
import tempfile
import wave

import numpy as np



CHANNELS = 1
MIN_RECORDING_SECONDS = 0.35
WHISPER_MODEL_NAME = "small"

# The English-only variants and turbo are intentionally excluded: this
# feature needs a multilingual model that supports speech-to-English translation.
SUPPORTED_TRANSLATION_MODELS = {"tiny", "base", "small", "medium", "large"}


@dataclass(frozen=True)
class SpeechResult:
    """English translation plus Whisper's automatically detected language."""

    english_text: str
    language_code: str
    language_name: str


def _validate_model_name(model_name: str) -> str:
    name = (model_name or WHISPER_MODEL_NAME).strip().lower()
    if name not in SUPPORTED_TRANSLATION_MODELS:
        choices = ", ".join(sorted(SUPPORTED_TRANSLATION_MODELS))
        raise ValueError(
            f"WHISPER_MODEL_NAME must be one of: {choices}. "
            "Use a multilingual model; 'turbo' is not suitable for translation."
        )
    return name


@lru_cache(maxsize=2)
def _load_whisper_model(model_name: str):
    """Load once and reuse the selected local Whisper model."""
    try:
        import whisper
    except ImportError as error:
        raise RuntimeError(
            "Voice input is optional and Whisper is not installed. "
            "Install requirements-voice.txt and the system packages in README.md."
        ) from error

    print(f"Loading local Whisper '{model_name}' model (first run may download it)...")
    return whisper.load_model(model_name)


def _language_name(language_code: str) -> str:
    if not language_code or language_code == "unknown":
        return "Unknown"
    try:
        from whisper.tokenizer import LANGUAGES

        return LANGUAGES.get(language_code, language_code.title())
    except Exception:
        return language_code.title()


def _default_input_sample_rate(sounddevice_module) -> int:
    """Return the native sample rate reported by the default input device."""
    try:
        device_info = sounddevice_module.query_devices(kind="input")
        sample_rate = int(round(float(device_info["default_samplerate"])))
    except Exception as error:
        raise RuntimeError(
            "No usable default microphone was found. Check the input device and permissions."
        ) from error

    if sample_rate <= 0:
        raise RuntimeError("The microphone reported an invalid sample rate.")
    return sample_rate


def record_microphone_audio() -> str:
    """Record mono audio at the default microphone's native rate.

    Returns a temporary WAV filename. The caller is responsible for deleting it.
    Whisper's audio loader resamples the WAV to its expected internal rate.
    """
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Microphone support is not installed. Install requirements-voice.txt."
        ) from error

    chunks = []
    callback_warnings = []

    # Use the selected microphone's native rate. Many devices support 44.1 or
    # 48 kHz but reject a forced 16 kHz stream. Whisper's audio loader resamples
    # the WAV to its internal 16 kHz format during transcription.
    recording_sample_rate = _default_input_sample_rate(sd)

    def audio_callback(indata, frames, time_info, status):
        if status:
            # Avoid printing from the real-time audio callback thread.
            callback_warnings.append(str(status))
        chunks.append(indata.copy())

    input("Press Enter when you are ready to speak...")
    print(
        f"Recording at {recording_sample_rate} Hz. "
        "Speak your command, then press Enter to stop."
    )
    try:
        with sd.InputStream(
            samplerate=recording_sample_rate,
            channels=CHANNELS,
            dtype="int16",
            callback=audio_callback,
        ):
            input()
    except Exception as error:
        raise RuntimeError(
            "Could not record from the microphone. Check microphone permissions, "
            "the selected input device, and PortAudio installation."
        ) from error

    for warning in callback_warnings:
        print(f"Microphone warning: {warning}")

    if not chunks:
        raise RuntimeError("No microphone audio was captured. Please try again.")

    audio = np.concatenate(chunks, axis=0)
    if audio.shape[0] < int(recording_sample_rate * MIN_RECORDING_SECONDS):
        raise RuntimeError("The recording was too short. Please speak for longer.")

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # int16 = 2 bytes per sample
            wav_file.setframerate(recording_sample_rate)
            wav_file.writeframes(audio.astype(np.int16, copy=False).tobytes())
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise

    return temp_path


def translate_audio_to_english(audio_path: str, model_name: Optional[str] = None) -> SpeechResult:
    """Detect the spoken language and return Whisper's English translation.

    `language=None` enables automatic language detection. The multilingual
    model's `translate` task produces English text for non-English speech.
    """
    selected_model = _validate_model_name(model_name or WHISPER_MODEL_NAME)
    model = _load_whisper_model(selected_model)

    try:
        result = model.transcribe(
            str(audio_path),
            task="translate",
            language=None,  # Ask Whisper to detect the spoken language.
            verbose=False,
        )
    except Exception as error:
        raise RuntimeError(f"Whisper could not process the recording: {error}") from error

    english_text = str(result.get("text", "")).strip()
    language_code = str(result.get("language", "unknown")).strip().lower()
    if not english_text:
        raise RuntimeError(
            "Whisper did not detect a clear command. Please reduce background noise and try again."
        )

    return SpeechResult(
        english_text=english_text,
        language_code=language_code,
        language_name=_language_name(language_code),
    )


def listen_and_translate() -> SpeechResult:
    """Record one microphone utterance, translate it, and remove the WAV file."""
    audio_path = record_microphone_audio()
    try:
        return translate_audio_to_english(audio_path)
    finally:
        Path(audio_path).unlink(missing_ok=True)
