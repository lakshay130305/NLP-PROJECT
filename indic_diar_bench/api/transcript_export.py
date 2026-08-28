"""Export formatters that work directly off a job's stored result dict (the same shape
`pipeline_runner.py::_build_result_dict` produces), since that's what's actually persisted
in the job store -- not the original dataclass objects, which don't survive past the
worker thread that produced them.

`to_txt` mirrors scripts/demo.py::format_transcript's layout (kept in sync manually, since
that function operates on live dataclass objects rather than this dict shape). `to_srt` is
a new format with no CLI equivalent.
"""

from __future__ import annotations


def to_txt(result: dict) -> str:
    duration = result.get("stats", {}).get("audio_duration", 0.0)
    lines = [f"Recording: {result['recording_id']}  ({duration:.1f}s)", "-" * 60]
    utterances = sorted(result["utterances"], key=lambda u: u["start"])
    for u in utterances:
        ts = f"[{u['start']:6.2f} - {u['end']:6.2f}]"
        lines.append(f"{ts}  {u['speaker']:<12} {u['text']}")
    if not utterances:
        lines.append("(no speech detected)")
    return "\n".join(lines)


def _srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    whole_secs = int(secs)
    millis = round((secs - whole_secs) * 1000)
    if millis == 1000:  # rounding carry
        millis = 0
        whole_secs += 1
    return f"{int(hours):02d}:{int(minutes):02d}:{whole_secs:02d},{millis:03d}"


def to_srt(result: dict) -> str:
    utterances = sorted(result["utterances"], key=lambda u: u["start"])
    lines: list[str] = []
    for i, u in enumerate(utterances, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(u['start'])} --> {_srt_timestamp(u['end'])}")
        lines.append(f"{u['speaker']}: {u['text']}")
        lines.append("")
    return "\n".join(lines)
