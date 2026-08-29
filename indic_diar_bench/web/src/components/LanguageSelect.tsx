import { useEffect, useState } from "react"
import { getLanguages } from "../api/client"
import type { LanguageOption } from "../api/types"

interface Props {
  value: string | null
  onChange: (value: string | null) => void
}

export function LanguageSelect({ value, onChange }: Props) {
  const [languages, setLanguages] = useState<LanguageOption[]>([])

  useEffect(() => {
    getLanguages()
      .then(setLanguages)
      .catch(() => setLanguages([]))
  }, [])

  return (
    <select
      className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm text-[var(--color-text)]"
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value || null)}
    >
      <option value="">Auto-detect language</option>
      {languages.map((lang) => (
        <option key={lang.name} value={lang.name}>
          {lang.name}
          {!lang.whisper_supported ? " (auto-detect fallback)" : ""}
        </option>
      ))}
    </select>
  )
}
