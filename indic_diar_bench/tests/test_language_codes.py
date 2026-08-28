from data.prep.language_codes import (
    LANGUAGE_TO_ISO639_1,
    language_to_whisper_code,
    whisper_supported_language_codes,
)
from data.prep.manifest import ALL_LANGUAGES


def test_every_dataset_language_has_an_entry():
    for lang in ALL_LANGUAGES:
        assert lang in LANGUAGE_TO_ISO639_1, f"missing ISO code for {lang}"


def test_supported_language_resolves_to_its_code():
    supported = whisper_supported_language_codes()
    resolved_count = sum(1 for lang in ALL_LANGUAGES if language_to_whisper_code(lang) is not None)
    assert resolved_count > 0  # at least some of the 22 languages must be Whisper-supported
    assert resolved_count <= len(ALL_LANGUAGES)
    for lang in ALL_LANGUAGES:
        code = language_to_whisper_code(lang)
        if code is not None:
            assert code in supported


def test_unknown_language_returns_none():
    assert language_to_whisper_code("Not A Real Language") is None


def test_unsupported_iso_code_falls_back_to_none():
    supported = whisper_supported_language_codes()
    unsupported_langs = [
        lang for lang, code in LANGUAGE_TO_ISO639_1.items() if code not in supported
    ]
    for lang in unsupported_langs:
        assert language_to_whisper_code(lang) is None
