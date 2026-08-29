import { FileAudio, WifiOff } from "lucide-react"
import { usePollJobs } from "../hooks/usePollJobs"
import { JobStatusBadge } from "./JobStatusBadge"

interface Props {
  onSelect: (jobId: string) => void
  selectedJobId: string | null
}

function relativeTime(iso?: string): string | null {
  if (!iso) return null
  const diffMs = Date.now() - new Date(iso).getTime()
  const diffSec = Math.max(0, Math.floor(diffMs / 1000))
  if (diffSec < 60) return "just now"
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.floor(diffMin / 60)
  return `${diffHr}h ago`
}

export function JobQueue({ onSelect, selectedJobId }: Props) {
  const { jobs, loading, error } = usePollJobs()

  if (loading && jobs.length === 0) {
    return <p className="text-sm text-[var(--color-text-muted)]">Loading jobs...</p>
  }

  if (jobs.length === 0) {
    if (error) {
      return (
        <div className="flex items-center gap-2 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
          <WifiOff size={16} />
          Can&apos;t reach the server. Retrying...
        </div>
      )
    }
    return (
      <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-[var(--color-border)] px-4 py-6 text-center text-sm text-[var(--color-text-muted)]">
        <FileAudio size={22} className="opacity-50" />
        No jobs yet. Upload an audio file to get started.
      </div>
    )
  }

  return (
    <ul className="flex flex-col gap-1.5">
      {jobs.map((job, i) => (
        <li key={job.job_id ?? `rejected-${i}-${job.filename}`}>
          <button
            disabled={!job.job_id}
            onClick={() => job.job_id && onSelect(job.job_id)}
            className={`flex w-full items-center justify-between gap-2 rounded-lg border px-3 py-2 text-left text-sm transition-colors ${
              selectedJobId === job.job_id
                ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]"
                : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/50"
            } ${!job.job_id ? "cursor-not-allowed opacity-60" : ""}`}
          >
            <span className="flex min-w-0 flex-col">
              <span className="truncate">{job.filename}</span>
              {relativeTime(job.created_at) && (
                <span className="text-xs text-[var(--color-text-muted)]">{relativeTime(job.created_at)}</span>
              )}
            </span>
            <JobStatusBadge status={job.status} />
          </button>
        </li>
      ))}
    </ul>
  )
}
