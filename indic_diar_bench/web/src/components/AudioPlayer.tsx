import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react"

export interface AudioPlayerHandle {
  seek: (time: number) => void
}

interface Props {
  src: string
  onTimeUpdate?: (time: number) => void
}

export const AudioPlayer = forwardRef<AudioPlayerHandle, Props>(function AudioPlayer(
  { src, onTimeUpdate },
  ref,
) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [duration, setDuration] = useState(0)

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
    const handleLoadedMetadata = () => setDuration(el.duration)
    el.addEventListener("timeupdate", handleTimeUpdate)
    el.addEventListener("loadedmetadata", handleLoadedMetadata)
    return () => {
      el.removeEventListener("timeupdate", handleTimeUpdate)
      el.removeEventListener("loadedmetadata", handleLoadedMetadata)
    }
  }, [onTimeUpdate])

  return (
    <audio ref={audioRef} src={src} controls className="w-full" data-duration={duration} />
  )
})
