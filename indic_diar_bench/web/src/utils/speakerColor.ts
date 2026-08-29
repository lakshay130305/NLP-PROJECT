export const SPEAKER_COLORS = [
  { chip: "bg-blue-500", panel: "bg-blue-100 text-blue-900 border-blue-300 dark:bg-blue-950 dark:text-blue-200 dark:border-blue-800" },
  { chip: "bg-purple-500", panel: "bg-purple-100 text-purple-900 border-purple-300 dark:bg-purple-950 dark:text-purple-200 dark:border-purple-800" },
  { chip: "bg-green-500", panel: "bg-green-100 text-green-900 border-green-300 dark:bg-green-950 dark:text-green-200 dark:border-green-800" },
  { chip: "bg-pink-500", panel: "bg-pink-100 text-pink-900 border-pink-300 dark:bg-pink-950 dark:text-pink-200 dark:border-pink-800" },
  { chip: "bg-orange-500", panel: "bg-orange-100 text-orange-900 border-orange-300 dark:bg-orange-950 dark:text-orange-200 dark:border-orange-800" },
  { chip: "bg-cyan-500", panel: "bg-cyan-100 text-cyan-900 border-cyan-300 dark:bg-cyan-950 dark:text-cyan-200 dark:border-cyan-800" },
  { chip: "bg-lime-500", panel: "bg-lime-100 text-lime-900 border-lime-300 dark:bg-lime-950 dark:text-lime-200 dark:border-lime-800" },
  { chip: "bg-fuchsia-500", panel: "bg-fuchsia-100 text-fuchsia-900 border-fuchsia-300 dark:bg-fuchsia-950 dark:text-fuchsia-200 dark:border-fuchsia-800" },
]

export function speakerColorIndex(speaker: string): number {
  let hash = 0
  for (let i = 0; i < speaker.length; i++) {
    hash = (hash * 31 + speaker.charCodeAt(i)) | 0
  }
  return Math.abs(hash) % SPEAKER_COLORS.length
}

export function colorForSpeaker(speaker: string) {
  return SPEAKER_COLORS[speakerColorIndex(speaker)]
}
