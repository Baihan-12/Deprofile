import { ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'

import PixelPanel from '@/components/PixelPanel'

type PromptPanelProps = {
  prompt: string
}

export default function PromptPanel({ prompt }: PromptPanelProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <PixelPanel
      title="System prompt"
      subtitle="The profile-grounded prompt currently sent to the model."
      action={
        <button className="pixel-btn" onClick={() => setExpanded((value) => !value)} type="button">
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      }
    >
      <pre className={`prompt-surface ${expanded ? 'max-h-[32rem]' : 'max-h-40'}`}>{prompt}</pre>
    </PixelPanel>
  )
}
