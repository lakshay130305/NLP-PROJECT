import { colorForSpeaker } from "../utils/speakerColor"

interface Props {
  speakers: string[]
  activeSpeaker: string | null
  onToggle: (speaker: string | null) => void
}

export function SpeakerLegend({ speakers, activeSpeaker, onToggle }: Props) {
  if (speakers.length === 0) return null

  return (
    <div className="flex flex-wrap gap-2">
      {speakers.map((speaker) => {
        const color = colorForSpeaker(speaker)
        const selected = activeSpeaker === speaker
        return (
          <button
            key={speaker}
            onClick={() => onToggle(selected ? null : speaker)}
            className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition-all ${
              selected
                ? "border-[var(--color-accent)] ring-1 ring-[var(--color-accent)]"
                : "border-[var(--color-border)] opacity-80 hover:opacity-100"
            } bg-[var(--color-surface)] text-[var(--color-text)]`}
          >
            <span className={`h-2 w-2 rounded-full ${color.chip}`} />
            {speaker}
          </button>
        )
      })}
    </div>
  )
}
