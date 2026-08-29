import { Loader2, UploadCloud, X } from "lucide-react"
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

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

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
          setFiles((prev) => [...prev, ...Array.from(e.dataTransfer.files)])
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
          dragOver
            ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]"
            : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/50"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".wav,.flac,.ogg,.aiff,.aif,.mp3,.m4a,.aac,.opus,.webm"
          className="hidden"
          onChange={(e) => setFiles((prev) => [...prev, ...Array.from(e.target.files ?? [])])}
        />
        <UploadCloud size={28} className="mx-auto mb-2 text-[var(--color-text-muted)]" />
        <p className="text-sm text-[var(--color-text-muted)]">
          {files.length > 0
            ? `${files.length} file${files.length === 1 ? "" : "s"} selected`
            : "Drop audio file(s) here, or click to browse"}
        </p>
      </div>

      {files.length > 0 && (
        <ul className="flex flex-col gap-1">
          {files.map((f, i) => (
            <li
              key={`${i}-${f.name}`}
              className="flex items-center justify-between rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 py-1.5 text-xs text-[var(--color-text)]"
            >
              <span className="truncate">{f.name}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  removeFile(i)
                }}
                className="ml-2 text-[var(--color-text-muted)] hover:text-red-600"
                aria-label={`Remove ${f.name}`}
              >
                <X size={14} />
              </button>
            </li>
          ))}
        </ul>
      )}

      <LanguageSelect value={language} onChange={setLanguage} />

      {error && <p className="text-sm text-red-700 dark:text-red-400">{error}</p>}

      <button
        onClick={submit}
        disabled={files.length === 0 || submitting}
        className="flex items-center justify-center gap-2 rounded-lg bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-50"
      >
        {submitting && <Loader2 size={14} className="animate-spin" />}
        {submitting ? "Uploading..." : "Transcribe"}
      </button>
    </div>
  )
}
