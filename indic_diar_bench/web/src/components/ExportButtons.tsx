import { exportUrl } from "../api/client"

export function ExportButtons({ jobId }: { jobId: string }) {
  const formats: Array<"txt" | "json" | "srt"> = ["txt", "json", "srt"]
  return (
    <div className="flex gap-2">
      {formats.map((fmt) => (
        <a
          key={fmt}
          href={exportUrl(jobId, fmt)}
          className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
        >
          Download .{fmt}
        </a>
      ))}
    </div>
  )
}
