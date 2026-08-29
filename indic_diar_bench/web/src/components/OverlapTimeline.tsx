import type { OverlapRegion } from "../api/types"

interface Props {
  regions: OverlapRegion[]
  duration: number
  onSeek: (time: number) => void
}

export function OverlapTimeline({ regions, duration, onSeek }: Props) {
  if (duration <= 0) return null

  return (
    <div className="mt-1">
      <div className="relative h-3 w-full overflow-hidden rounded bg-[var(--color-border)]">
        {regions.map((r, i) => {
          const left = (r.start / duration) * 100
          const width = Math.max(0.5, ((r.end - r.start) / duration) * 100)
          return (
            <div
              key={i}
              title={`Overlap ${r.start.toFixed(1)}s-${r.end.toFixed(1)}s: ${r.speakers.join(", ")}`}
              className="absolute top-0 h-full cursor-pointer bg-amber-400/70 transition-colors hover:bg-amber-500"
              style={{ left: `${left}%`, width: `${width}%` }}
              onClick={() => onSeek(r.start)}
            />
          )
        })}
      </div>
      {regions.length > 0 && (
        <p className="mt-1 text-xs text-[var(--color-text-muted)]">
          {regions.length} overlapping speech region{regions.length === 1 ? "" : "s"} detected (amber bands above)
        </p>
      )}
    </div>
  )
}
