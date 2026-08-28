import type { JobStatus } from "../api/types"

const STYLES: Record<JobStatus, string> = {
  queued: "bg-slate-200 text-slate-700",
  processing: "bg-amber-200 text-amber-800",
  done: "bg-emerald-200 text-emerald-800",
  failed: "bg-red-200 text-red-800",
  rejected: "bg-red-200 text-red-800",
}

export function JobStatusBadge({ status }: { status: JobStatus }) {
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${STYLES[status]}`}>
      {status}
    </span>
  )
}
