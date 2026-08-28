import type { Utterance } from "../api/types"

const SPEAKER_COLORS = [
  "bg-blue-100 text-blue-900 border-blue-300",
  "bg-purple-100 text-purple-900 border-purple-300",
  "bg-green-100 text-green-900 border-green-300",
  "bg-pink-100 text-pink-900 border-pink-300",
  "bg-orange-100 text-orange-900 border-orange-300",
  "bg-cyan-100 text-cyan-900 border-cyan-300",
  "bg-lime-100 text-lime-900 border-lime-300",
  "bg-fuchsia-100 text-fuchsia-900 border-fuchsia-300",
]

function colorForSpeaker(speaker: string): string {
  let hash = 0
  for (let i = 0; i < speaker.length; i++) {
    hash = (hash * 31 + speaker.charCodeAt(i)) | 0
  }
  const index = Math.abs(hash) % SPEAKER_COLORS.length
  return SPEAKER_COLORS[index]
}

interface Props {
  utterances: Utterance[]
  currentTime: number
  onSeek: (time: number) => void
}

export function UtteranceList({ utterances, currentTime, onSeek }: Props) {
  if (utterances.length === 0) {
    return <p className="text-sm text-slate-500 italic">No speech detected.</p>
  }

  return (
    <div className="flex flex-col gap-2">
      {utterances.map((u, i) => {
        const active = currentTime >= u.start && currentTime < u.end
        return (
          <button
            key={i}
            onClick={() => onSeek(u.start)}
            className={`text-left rounded-lg border px-3 py-2 transition-shadow ${colorForSpeaker(u.speaker)} ${
              active ? "ring-2 ring-offset-1 ring-slate-900" : ""
            }`}
          >
            <div className="flex items-baseline gap-2 text-xs opacity-70">
              <span className="font-mono">
                {u.start.toFixed(2)}s - {u.end.toFixed(2)}s
              </span>
              <span className="font-semibold">{u.speaker}</span>
            </div>
            <p className="mt-0.5 text-sm">{u.text}</p>
          </button>
        )
      })}
    </div>
  )
}
