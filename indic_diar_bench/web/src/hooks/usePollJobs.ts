import { useEffect, useRef, useState } from "react"
import { listJobs } from "../api/client"
import type { JobSummary } from "../api/types"

const ACTIVE_STATUSES = new Set(["queued", "processing"])

export function usePollJobs(intervalMs = 3000) {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const jobsRef = useRef<JobSummary[]>([])

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    async function tick() {
      try {
        const list = await listJobs()
        if (cancelled) return
        jobsRef.current = list
        setJobs(list)
        setError(null)
        setLoading(false)
      } catch (e) {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      }
      if (cancelled) return
      const anyActive = jobsRef.current.some((j) => ACTIVE_STATUSES.has(j.status))
      timer = window.setTimeout(tick, anyActive ? intervalMs : intervalMs * 2)
    }

    tick()
    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [intervalMs])

  return { jobs, loading, error }
}
