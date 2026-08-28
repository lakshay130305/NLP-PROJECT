import { useRef, useState } from "react"
import { createJobs } from "../api/client"
import { LanguageSelect } from "./LanguageSelect"

interface Props {
  onUploaded: (firstJobId: string | null) => void
}

export function UploadView({ onUploaded }: Props) {
  const [files, setFiles] = useState<File[]>([])
  const [language, setLanguage] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  async function submit() {
    if (files.length === 0) return
    setSubmitting(true)
    setError(null)
    try {
      const jobs = await createJobs(files, language)
      setFiles([])
      const firstReal = jobs.find((j) => j.job_id)?.job_id ?? null
      onUploaded(firstReal)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          setFiles(Array.from(e.dataTransfer.files))
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
          dragOver ? "border-slate-900 bg-slate-100" : "border-slate-300 bg-white"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".wav,.flac,.ogg,.aiff,.aif,.mp3,.m4a,.aac,.opus,.webm"
          className="hidden"
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
        />
        <p className="text-sm text-slate-600">
          {files.length > 0
            ? `${files.length} file${files.length === 1 ? "" : "s"} selected`
            : "Drop audio file(s) here, or click to browse"}
        </p>
        {files.length > 0 && (
          <ul className="mt-2 text-xs text-slate-500">
            {files.map((f) => (
              <li key={f.name}>{f.name}</li>
            ))}
          </ul>
        )}
      </div>

      <LanguageSelect value={language} onChange={setLanguage} />

      {error && <p className="text-sm text-red-700">{error}</p>}

      <button
        onClick={submit}
        disabled={files.length === 0 || submitting}
        className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
      >
        {submitting ? "Uploading..." : "Transcribe"}
      </button>
    </div>
  )
}
