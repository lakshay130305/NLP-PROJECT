import type { Utterance, WordToken } from "../api/types"
import { colorForSpeaker } from "../utils/speakerColor"

interface Props {
  utterances: Utterance[]
  currentTime: number
  onSeek: (time: number) => void
  activeSpeaker: string | null
}

function confidenceClass(confidence: number): string {
  if (confidence < 0.4) return "underline decoration-red-500 decoration-2 decoration-wavy"
  if (confidence < 0.6) return "underline decoration-amber-500 decoration-2"
  return ""
}

function Word({ word, active }: { word: WordToken; active: boolean }) {
  return (
    <span
      title={`confidence: ${(word.confidence * 100).toFixed(0)}%`}
      className={`${confidenceClass(word.confidence)} ${active ? "rounded bg-[var(--color-accent)]/25" : ""}`}
    >
      {word.text}
    </span>
  )
}

export function UtteranceList({ utterances, currentTime, onSeek, activeSpeaker }: Props) {
  if (utterances.length === 0) {
    return <p className="text-sm italic text-[var(--color-text-muted)]">No speech detected.</p>
  }

  return (
    <div className="flex flex-col gap-2">
      {utterances.map((u, i) => {
        const active = currentTime >= u.start && currentTime < u.end
        const dimmed = activeSpeaker !== null && activeSpeaker !== u.speaker
        const color = colorForSpeaker(u.speaker)
        return (
          <div
            key={i}
            onClick={() => onSeek(u.start)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && onSeek(u.start)}
            className={`cursor-pointer rounded-lg border px-3 py-2 text-left transition-all ${color.panel} ${
              active ? "ring-2 ring-offset-1 ring-[var(--color-accent)]" : ""
            } ${dimmed ? "opacity-30" : "opacity-100"}`}
          >
            <div className="flex items-baseline gap-2 text-xs opacity-70">
              <span className="font-mono">
                {u.start.toFixed(2)}s - {u.end.toFixed(2)}s
              </span>
              <span className="font-semibold">{u.speaker}</span>
            </div>
            <p className="mt-0.5 text-sm leading-relaxed">
              {u.words.length > 0
                ? u.words.map((w, wi) => (
                    <span key={wi}>
                      <Word word={w} active={active && currentTime >= w.start && currentTime < w.end} />
                      {wi < u.words.length - 1 ? " " : ""}
                    </span>
                  ))
                : u.text}
            </p>
          </div>
        )
      })}
    </div>
  )
}
