import { AlertTriangle, Clock, Gauge, Layers, Users, WifiOff } from "lucide-react"
import { useMemo, useRef, useState } from "react"
import { audioUrl } from "../api/client"
import type { JobDetail } from "../api/types"
import { AudioPlayer, type AudioPlayerHandle } from "./AudioPlayer"
import { ExportButtons } from "./ExportButtons"
import { JobStatusBadge } from "./JobStatusBadge"
import { OverlapTimeline } from "./OverlapTimeline"
import { SpeakerLegend } from "./SpeakerLegend"
import { UtteranceList } from "./UtteranceList"

export function TranscriptView({ job, connectionError }: { job: JobDetail; connectionError?: string | null }) {
  const playerRef = useRef<AudioPlayerHandle>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [activeSpeaker, setActiveSpeaker] = useState<string | null>(null)

  const seek = (time: number) => playerRef.current?.seek(time)

  const speakers = useMemo(
    () => (job.result ? [...new Set(job.result.utterances.map((u) => u.speaker))].sort() : []),
    [job.result],
  )

  return (
    <div className="flex flex-col gap-4 animate-fade-in">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-[var(--color-text)]">{job.filename}</h2>
        <JobStatusBadge status={job.status} />
      </div>

      {connectionError && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-300 bg-amber-50 px-4 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
          <WifiOff size={14} />
          Lost connection to server, retrying...
        </div>
      )}

      {(job.status === "failed" || job.status === "rejected") && (
        <div className="flex items-center gap-2 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
          <AlertTriangle size={16} />
          {job.error ?? "Processing failed."}
        </div>
      )}

      {(job.status === "queued" || job.status === "processing") && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
          {job.status === "queued" ? "Waiting in queue..." : "Processing... this can take a few minutes."}
        </div>
      )}

      {job.status === "done" && job.result && (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat icon={Clock} label="Duration" value={`${job.result.stats.audio_duration.toFixed(1)}s`} />
            <Stat icon={Users} label="Speakers" value={String(job.result.stats.speaker_count)} />
            <Stat icon={Gauge} label="Real-time factor" value={`${job.result.stats.rtf.toFixed(2)}x`} />
            <Stat
              icon={Layers}
              label="Overlap"
              value={`${(
                (job.result.overlap_regions.reduce((sum, r) => sum + (r.end - r.start), 0) /
                  Math.max(job.result.stats.audio_duration, 1e-6)) *
                100
              ).toFixed(1)}%`}
            />
          </div>

          <div>
            <AudioPlayer ref={playerRef} src={audioUrl(job.job_id)} onTimeUpdate={setCurrentTime} />
            <OverlapTimeline
              regions={job.result.overlap_regions}
              duration={job.result.stats.audio_duration}
              onSeek={seek}
            />
          </div>

          <SpeakerLegend speakers={speakers} activeSpeaker={activeSpeaker} onToggle={setActiveSpeaker} />

          <ExportButtons jobId={job.job_id} utterances={job.result.utterances} />

          <UtteranceList
            utterances={job.result.utterances}
            currentTime={currentTime}
            onSeek={seek}
            activeSpeaker={activeSpeaker}
          />
        </>
      )}
    </div>
  )
}

function Stat({ icon: Icon, label, value }: { icon: typeof Clock; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2">
      <div className="rounded-md bg-[var(--color-accent-soft)] p-1.5 text-[var(--color-accent)]">
        <Icon size={16} />
      </div>
      <div>
        <p className="text-xs text-[var(--color-text-muted)]">{label}</p>
        <p className="text-lg font-semibold text-[var(--color-text)]">{value}</p>
      </div>
    </div>
  )
}
