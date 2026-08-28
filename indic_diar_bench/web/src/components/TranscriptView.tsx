import { useRef, useState } from "react"
import { audioUrl } from "../api/client"
import type { JobDetail } from "../api/types"
import { AudioPlayer, type AudioPlayerHandle } from "./AudioPlayer"
import { ExportButtons } from "./ExportButtons"
import { JobStatusBadge } from "./JobStatusBadge"
import { OverlapTimeline } from "./OverlapTimeline"
import { UtteranceList } from "./UtteranceList"

export function TranscriptView({ job }: { job: JobDetail }) {
  const playerRef = useRef<AudioPlayerHandle>(null)
  const [currentTime, setCurrentTime] = useState(0)

  const seek = (time: number) => playerRef.current?.seek(time)

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">{job.filename}</h2>
        <JobStatusBadge status={job.status} />
      </div>

      {job.status === "failed" && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
          {job.error ?? "Processing failed."}
        </div>
      )}

      {(job.status === "queued" || job.status === "processing") && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          {job.status === "queued" ? "Waiting in queue..." : "Processing... this can take a few minutes."}
        </div>
      )}

      {job.status === "done" && job.result && (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="Duration" value={`${job.result.stats.audio_duration.toFixed(1)}s`} />
            <Stat label="Speakers" value={String(job.result.stats.speaker_count)} />
            <Stat label="Real-time factor" value={`${job.result.stats.rtf.toFixed(2)}x`} />
            <Stat
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

          <ExportButtons jobId={job.job_id} />

          <UtteranceList utterances={job.result.utterances} currentTime={currentTime} onSeek={seek} />
        </>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-lg font-semibold text-slate-900">{value}</p>
    </div>
  )
}
