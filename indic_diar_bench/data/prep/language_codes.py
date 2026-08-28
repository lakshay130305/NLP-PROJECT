"""Maps Indic DiarBench's language names (ALL_LANGUAGES in manifest.py) to ISO 639-1
codes, for passing the recording's KNOWN language into ASR as a hint instead of relying
on blind auto-detection.

Not every one of the 22 scheduled languages is supported by Whisper/faster-whisper --
Whisper (as of the checkpoints used here) covers 100 languages total, of which 14 of our
22 overlap. The other 8 (commercially/academically lower-resource even by Indian-language
standards -- Bodo, Dogri, Kashmiri, Konkani, Maithili, Manipuri, Odia, Santali) have no
Whisper code and fall back to auto-detection (None), which is unlikely to work well for
them either, but is the best available option without a language-specific ASR backend.

`whisper_supported_language_codes()` reads faster_whisper's own supported-code set at
call time rather than hardcoding an assumed list, so this module can never silently claim
support for a code that isn't actually in the installed faster-whisper version.
"""

from __future__ import annotations

# ISO 639-1 codes for the 22 languages in data/prep/manifest.py::ALL_LANGUAGES.
# Source: ISO 639-1 standard codes for each language name (independent of what any
# particular ASR backend happens to support -- see `language_to_whisper_code` below for
# the backend-support-checked lookup actually used by the pipeline).
LANGUAGE_TO_ISO639_1: dict[str, str] = {
    "Assamese": "as",
    "Bengali": "bn",
    "Bodo": "brx",  # not a 2-letter ISO 639-1 code; Bodo has no ISO 639-1 assignment
    "Dogri": "doi",  # ISO 639-2 only, no ISO 639-1 assignment
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Kashmiri": "ks",
    "Konkani": "kok",  # ISO 639-2 only, no ISO 639-1 assignment
    "Maithili": "mai",  # ISO 639-2 only, no ISO 639-1 assignment
    "Malayalam": "ml",
    "Manipuri": "mni",  # ISO 639-2 only (Meitei/Manipuri), no ISO 639-1 assignment
    "Marathi": "mr",
    "Nepali": "ne",
    "Odia": "or",
    "Punjabi": "pa",
    "Sanskrit": "sa",
    "Santali": "sat",  # ISO 639-2 only, no ISO 639-1 assignment
    "Sindhi": "sd",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
}


def whisper_supported_language_codes() -> set[str]:
    """The actual set of language codes the installed faster-whisper build supports --
    read live from the library rather than assumed, so this stays correct across versions."""
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    return set(_LANGUAGE_CODES)


def language_to_whisper_code(language_name: str) -> str | None:
    """Returns a Whisper language code for `language_name` if BOTH (a) it has a known
    ISO 639-1 code and (b) that code is actually supported by the installed faster-whisper
    build. Returns None otherwise, meaning "let Whisper auto-detect" -- the correct
    fallback for a language it doesn't have a dedicated code for."""
    code = LANGUAGE_TO_ISO639_1.get(language_name)
    if code is None:
        return None
    if code not in whisper_supported_language_codes():
        return None
    return code
