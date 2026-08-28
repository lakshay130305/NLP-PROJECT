export type JobStatus = "queued" | "processing" | "done" | "failed" | "rejected"

export interface JobSummary {
  job_id: string | null
  filename: string
  status: JobStatus
  created_at?: string
  language_hint?: string | null
  error?: string
}

export interface WordToken {
  text: string
  start: number
  end: number
  confidence: number
}

export interface Utterance {
  speaker: string
  start: number
  end: number
  text: string
  words: WordToken[]
}

export interface OverlapRegion {
  start: number
  end: number
  speakers: string[]
}

export interface TranscriptStats {
  audio_duration: number
  wall_clock_time: number
  rtf: number
  speaker_count: number
}

export interface TranscriptResult {
  recording_id: string
  utterances: Utterance[]
  stats: TranscriptStats
  overlap_regions: OverlapRegion[]
}

export interface JobDetail {
  job_id: string
  filename: string
  status: JobStatus
  created_at: string
  started_at: string | null
  finished_at: string | null
  language_hint: string | null
  error: string | null
  result: TranscriptResult | null
}

export interface LanguageOption {
  name: string
  iso639_1: string
  whisper_supported: boolean
}
