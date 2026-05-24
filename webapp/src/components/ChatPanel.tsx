import { AlertTriangle, SendHorizonal, Sparkles, Trash2 } from 'lucide-react'
import { FormEvent, useState } from 'react'

import PixelPanel from '@/components/PixelPanel'
import type { ChatTurn } from '@/utils/types'

type ChatPanelProps = {
  messages: ChatTurn[]
  disabled: boolean
  sending: boolean
  error?: string
  onReset: () => void
  onSend: (content: string) => Promise<void>
}

export default function ChatPanel({ messages, disabled, sending, error, onReset, onSend }: ChatPanelProps) {
  const [draft, setDraft] = useState('')

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!draft.trim()) {
      return
    }
    const content = draft
    setDraft('')
    await onSend(content)
  }

  return (
    <PixelPanel
      title="Patient chat"
      subtitle="Each turn is sent with the selected profile and the current system prompt."
      action={
        <button className="pixel-btn" onClick={onReset} type="button">
          <Trash2 className="h-4 w-4" />
          Clear chat
        </button>
      }
    >
      <div className="chat-grid">
        <div className="chat-stream">
          {messages.length ? (
            messages.map((message, index) => (
              <article className={`chat-bubble ${message.role === 'assistant' ? 'chat-bubble-assistant' : 'chat-bubble-user'}`} key={`${message.role}-${index}`}>
                <p className="mb-2 flex items-center gap-2 text-[11px] uppercase tracking-[0.25em] text-violet-100/50">
                  <Sparkles className="h-4 w-4" />
                  {message.role === 'assistant' ? 'Simulated patient' : 'Researcher'}
                </p>
                <p className="text-sm leading-7 text-violet-50/90">{message.content}</p>
              </article>
            ))
          ) : (
            <div className="empty-chat-state">
              <Sparkles className="h-6 w-6 text-cyan-200" />
              <p>You can start with a question like: “When did things begin to feel worse for you?”</p>
            </div>
          )}
        </div>
        {error ? (
          <div className="mt-4 rounded-xl border border-pink-400/50 bg-pink-500/10 p-4 text-sm text-pink-100">
            <p className="flex items-center gap-2 font-medium text-pink-200">
              <AlertTriangle className="h-4 w-4" />
              Request failed
            </p>
            <p className="mt-2 leading-6">{error}</p>
          </div>
        ) : null}
        <form className="mt-4 grid gap-3" onSubmit={handleSubmit}>
          <textarea
            className="pixel-textarea"
            disabled={disabled || sending}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Type a question for the patient..."
            value={draft}
          />
          <div className="flex flex-wrap justify-end gap-3">
            <button className="pixel-btn pixel-btn-primary" disabled={disabled || sending} type="submit">
              <SendHorizonal className="h-4 w-4" />
              {sending ? 'Sending...' : 'Send question'}
            </button>
          </div>
        </form>
      </div>
    </PixelPanel>
  )
}
