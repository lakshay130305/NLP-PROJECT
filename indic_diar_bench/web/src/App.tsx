import { useEffect, useState } from "react"
import { getHealth } from "./api/client"
import { JobQueue } from "./components/JobQueue"
import { TranscriptView } from "./components/TranscriptView"
import { UploadView } from "./components/UploadView"
import { usePollJob } from "./hooks/usePollJob"

export default function App() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [pipelineError, setPipelineError] = useState<string | null>(null)
  const { job } = usePollJob(selectedJobId)

  useEffect(() => {
    getHealth()
      .then((h) => setPipelineError(h.pipeline_load_error))
      .catch(() => {})
  }, [])

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white px-6 py-4">
        <h1 className="text-xl font-bold text-slate-900">Indic DiarBench &mdash; Speaker-Attributed Transcription</h1>
        <p className="text-sm text-slate-500">Overlap-aware diarization + multilingual ASR demo</p>
      </header>

      {pipelineError && (
        <div className="bg-red-50 px-6 py-2 text-sm text-red-800">
          Server warning: the transcription pipeline failed to load ({pipelineError}). Jobs will fail until this is
          resolved.
        </div>
      )}

      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-6 p-6 md:grid-cols-[320px_1fr]">
        <aside className="flex flex-col gap-6">
          <section>
            <h2 className="mb-2 text-sm font-semibold text-slate-700">Upload</h2>
            <UploadView onUploaded={setSelectedJobId} />
          </section>
          <section>
            <h2 className="mb-2 text-sm font-semibold text-slate-700">Jobs</h2>
            <JobQueue onSelect={setSelectedJobId} selectedJobId={selectedJobId} />
          </section>
        </aside>

        <section className="rounded-xl border border-slate-200 bg-white p-6">
          {job ? (
            <TranscriptView job={job} />
          ) : (
            <p className="text-sm text-slate-500">Select a job from the list, or upload a new file, to see its transcript.</p>
          )}
        </section>
      </main>
    </div>
  )
}
