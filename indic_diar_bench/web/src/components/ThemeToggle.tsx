import { Moon, Sun } from "lucide-react"
import { useEffect, useState } from "react"

type Theme = "light" | "dark"

function getInitialTheme(): Theme {
  try {
    const stored = localStorage.getItem("theme")
    if (stored === "light" || stored === "dark") return stored
  } catch {
    // localStorage unavailable, fall through to system preference
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme)

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme)
    try {
      localStorage.setItem("theme", theme)
    } catch {
      // ignore write failures (private browsing, etc.)
    }
  }, [theme])

  return (
    <button
      onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
      aria-label="Toggle dark mode"
      className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-2 text-[var(--color-text-muted)] transition-colors hover:text-[var(--color-text)]"
    >
      {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  )
}
