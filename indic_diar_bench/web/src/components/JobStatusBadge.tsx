import { CheckCircle2, Clock, Loader2, XCircle } from "lucide-react"
import type { JobStatus } from "../api/types"

const STYLES: Record<JobStatus, string> = {
  queued: "bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  processing: "bg-amber-200 text-amber-800 dark:bg-amber-900 dark:text-amber-300",
  done: "bg-emerald-200 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-300",
  failed: "bg-red-200 text-red-800 dark:bg-red-900 dark:text-red-300",
  rejected: "bg-red-200 text-red-800 dark:bg-red-900 dark:text-red-300",
}

const ICONS: Record<JobStatus, typeof Clock> = {
  queued: Clock,
  processing: Loader2,
  done: CheckCircle2,
  failed: XCircle,
  rejected: XCircle,
}

export function JobStatusBadge({ status }: { status: JobStatus }) {
  const Icon = ICONS[status]
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${STYLES[status]}`}>
      <Icon size={12} className={status === "processing" ? "animate-spin" : ""} />
      {status}
    </span>
  )
}
