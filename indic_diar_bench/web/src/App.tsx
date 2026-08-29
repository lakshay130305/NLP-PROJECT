import { AudioWaveform, FileAudio } from "lucide-react"
import { useEffect, useState } from "react"
import { getHealth } from "./api/client"
import { JobQueue } from "./components/JobQueue"
import { TranscriptSkeleton } from "./components/Skeleton"
import { ThemeToggle } from "./components/ThemeToggle"
import { TranscriptView } from "./components/TranscriptView"
import { UploadView } from "./components/UploadView"
import { usePollJob } from "./hooks/usePollJob"

export default function App() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const { job, error } = usePollJob(selectedJobId)
  const switching = selectedJobId !== null && (job === null || job.job_id !== selectedJobId)

  useEffect(() => {
    getHealth()
      .then((h) => setPipelineError(h.pipeline_load_error))
      .catch(() => {})
  }, [])

  return (
    <div className="min-h-screen bg-[var(--color-bg)]">
      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-[var(--color-border)] bg-[var(--color-surface)]/90 px-6 py-4 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-[var(--color-accent-soft)] p-2 text-[var(--color-accent)]">
            <AudioWaveform size={20} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-[var(--color-text)]">Indic DiarBench</h1>
            <p className="text-sm text-[var(--color-text-muted)]">Overlap-aware diarization + multilingual ASR demo</p>
          </div>
        </div>
        <ThemeToggle />
      </header>

      {pipelineError && (
        <div className="bg-red-50 px-6 py-2 text-sm text-red-800 dark:bg-red-950 dark:text-red-300">
          Server warning: the transcription pipeline failed to load ({pipelineError}). Jobs will fail until this is
          resolved.
        </div>
      )}

      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-6 p-6 md:grid-cols-[320px_1fr]">
        <aside className="flex flex-col gap-6">
          <section>
            <h2 className="mb-2 text-sm font-semibold text-[var(--color-text)]">Upload</h2>
            <UploadView onUploaded={setSelectedJobId} />
          </section>
          <section>
            <h2 className="mb-2 text-sm font-semibold text-[var(--color-text)]">Jobs</h2>
            <JobQueue onSelect={setSelectedJobId} selectedJobId={selectedJobId} />
          </section>
        </aside>

        <section className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
          {switching ? (
            <TranscriptSkeleton />
          ) : job ? (
            <TranscriptView job={job} connectionError={error} />
          ) : (
            <div className="flex flex-col items-center gap-2 py-16 text-center text-[var(--color-text-muted)]">
              <FileAudio size={32} className="opacity-40" />
              <p className="text-sm">Select a job from the list, or upload a new file, to see its transcript.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
