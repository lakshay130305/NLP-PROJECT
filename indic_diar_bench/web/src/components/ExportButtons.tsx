import { Check, Copy, Download } from "lucide-react"
import { useState } from "react"
import { exportUrl } from "../api/client"
import type { Utterance } from "../api/types"

function formatTranscript(utterances: Utterance[]): string {
  return utterances
    .map((u) => `[${u.start.toFixed(2)}s - ${u.end.toFixed(2)}s] ${u.speaker}: ${u.text}`)
    .join("\n")
}

export function ExportButtons({ jobId, utterances }: { jobId: string; utterances: Utterance[] }) {
  const [copied, setCopied] = useState(false)
  const formats: Array<"txt" | "json" | "srt"> = ["txt", "json", "srt"]

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(formatTranscript(utterances))
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      // clipboard permission denied; silently ignore
    }
  }

  return (
    <div className="flex flex-wrap gap-2">
      <button
        onClick={handleCopy}
        className="flex items-center gap-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5 text-xs font-medium text-[var(--color-text)] transition-colors hover:border-[var(--color-accent)]"
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
        {copied ? "Copied" : "Copy transcript"}
      </button>
      {formats.map((fmt) => (
        <a
          key={fmt}
          href={exportUrl(jobId, fmt)}
          className="flex items-center gap-1.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5 text-xs font-medium text-[var(--color-text)] transition-colors hover:border-[var(--color-accent)]"
        >
          <Download size={14} />.{fmt}
        </a>
      ))}
    </div>
  )
}
