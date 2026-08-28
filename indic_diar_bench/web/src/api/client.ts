import type { JobDetail, JobSummary, LanguageOption } from "./types"

async function jsonOrThrow(resp: Response) {
  if (!resp.ok) {
    let detail = resp.statusText
    try {
      const body = await resp.json()
      detail = body.detail ?? detail
    } catch {
      // response wasn't JSON, keep statusText
    }
    throw new Error(`${resp.status}: ${detail}`)
  }
  return resp.json()
}

export async function createJobs(files: File[], language: string | null): Promise<JobSummary[]> {
  const form = new FormData()
  for (const file of files) {
    form.append("files", file)
  }
  if (language) {
    form.append("language", language)
  }
  const resp = await fetch("/api/jobs", { method: "POST", body: form })
  const body = await jsonOrThrow(resp)
  return body.jobs
}

export async function listJobs(): Promise<JobSummary[]> {
  const resp = await fetch("/api/jobs")
  const body = await jsonOrThrow(resp)
  return body.jobs
}

export async function getJob(jobId: string): Promise<JobDetail> {
  const resp = await fetch(`/api/jobs/${jobId}`)
  return jsonOrThrow(resp)
}

export async function getLanguages(): Promise<LanguageOption[]> {
  const resp = await fetch("/api/languages")
  const body = await jsonOrThrow(resp)
  return body.languages
}

export async function getHealth(): Promise<{ status: string; pipeline_loaded: boolean; pipeline_load_error: string | null }> {
  const resp = await fetch("/api/health")
  return jsonOrThrow(resp)
}

export function audioUrl(jobId: string): string {
  return `/api/jobs/${jobId}/audio`
}

export function exportUrl(jobId: string, fmt: "txt" | "json" | "srt"): string {
  return `/api/jobs/${jobId}/export/${fmt}`
}
