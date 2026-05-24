import type { ReactNode } from 'react'

type PixelPanelProps = {
  title: string
  subtitle?: string
  action?: ReactNode
  className?: string
  children: ReactNode
}

export default function PixelPanel({ title, subtitle, action, className = '', children }: PixelPanelProps) {
  return (
    <section className={`pixel-panel ${className}`.trim()}>
      <header className="mb-4 flex items-start justify-between gap-4 border-b border-white/10 pb-3">
        <div>
          <p className="pixel-label">{title}</p>
          {subtitle ? <p className="mt-2 text-sm text-violet-100/70">{subtitle}</p> : null}
        </div>
        {action}
      </header>
      {children}
    </section>
  )
}
