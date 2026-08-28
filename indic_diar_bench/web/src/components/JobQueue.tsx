import { usePollJobs } from "../hooks/usePollJobs"
import { JobStatusBadge } from "./JobStatusBadge"

interface Props {
  onSelect: (jobId: string) => void
  selectedJobId: string | null
}

export function JobQueue({ onSelect, selectedJobId }: Props) {
  const { jobs, loading } = usePollJobs()

  if (loading && jobs.length === 0) {
    return <p className="text-sm text-slate-500">Loading jobs...</p>
  }

  if (jobs.length === 0) {
    return <p className="text-sm text-slate-500">No jobs yet. Upload an audio file to get started.</p>
  }

  return (
    <ul className="flex flex-col gap-1.5">
      {jobs.map((job) => (
        <li key={job.job_id ?? job.filename}>
          <button
            disabled={!job.job_id}
            onClick={() => job.job_id && onSelect(job.job_id)}
            className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left text-sm transition-colors ${
              selectedJobId === job.job_id ? "border-slate-900 bg-slate-100" : "border-slate-200 bg-white hover:bg-slate-50"
            } ${!job.job_id ? "cursor-not-allowed opacity-60" : ""}`}
          >
            <span className="truncate">{job.filename}</span>
            <JobStatusBadge status={job.status} />
          </button>
        </li>
      ))}
    </ul>
  )
}
