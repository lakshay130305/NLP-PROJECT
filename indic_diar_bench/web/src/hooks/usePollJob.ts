import { useEffect, useState } from "react"
import { getJob } from "../api/client"
import type { JobDetail } from "../api/types"

const TERMINAL_STATUSES = new Set(["done", "failed"])

export function usePollJob(jobId: string | null, intervalMs = 1500) {
  const [job, setJob] = useState<JobDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    let timer: number | undefined

    async function tick() {
      try {
        const j = await getJob(jobId!)
        if (cancelled) return
        setJob(j)
        setError(null)
        if (!TERMINAL_STATUSES.has(j.status)) {
          timer = window.setTimeout(tick, intervalMs)
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      }
    }

    tick()
    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [jobId, intervalMs])

  return { job, error }
}
