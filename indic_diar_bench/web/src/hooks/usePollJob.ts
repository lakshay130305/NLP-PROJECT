import { useEffect, useState } from "react"
import { getJob } from "../api/client"
import type { JobDetail } from "../api/types"

const TERMINAL_STATUSES = new Set(["done", "failed", "rejected"])

export function usePollJob(jobId: string | null, intervalMs = 1500) {
  const [job, setJob] = useState<JobDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setJob(null)
    setError(null)
    if (!jobId) return
    let cancelled = false
    let timer: number | undefined

    async function tick() {
      let status: string | undefined
      try {
        const j = await getJob(jobId!)
        if (cancelled) return
        setJob(j)
        setError(null)
        status = j.status
      } catch (e) {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
      }
      if (cancelled) return
      if (!status || !TERMINAL_STATUSES.has(status)) {
        timer = window.setTimeout(tick, intervalMs)
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
