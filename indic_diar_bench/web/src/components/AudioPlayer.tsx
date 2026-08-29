import { forwardRef, useEffect, useImperativeHandle, useRef } from "react"

export interface AudioPlayerHandle {
  seek: (time: number) => void
}

interface Props {
  src: string
  onTimeUpdate?: (time: number) => void
}

function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false
  const tag = el.tagName
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable
}

export const AudioPlayer = forwardRef<AudioPlayerHandle, Props>(function AudioPlayer(
  { src, onTimeUpdate },
  ref,
) {
  const audioRef = useRef<HTMLAudioElement>(null)

  useImperativeHandle(ref, () => ({
    seek(time: number) {
      const el = audioRef.current
      if (!el) return
      el.currentTime = time
      void el.play()
    },
  }))

  useEffect(() => {
    const el = audioRef.current
    if (!el) return
    const handleTimeUpdate = () => onTimeUpdate?.(el.currentTime)
    el.addEventListener("timeupdate", handleTimeUpdate)
    return () => el.removeEventListener("timeupdate", handleTimeUpdate)
  }, [onTimeUpdate])

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.code !== "Space" || isTypingTarget(e.target)) return
      const el = audioRef.current
      if (!el) return
      e.preventDefault()
      if (el.paused) void el.play()
      else el.pause()
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [])

  return <audio ref={audioRef} src={src} controls className="w-full" />
})
