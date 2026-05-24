import { LoaderCircle, Radio, ShieldCheck } from 'lucide-react'

import PixelPanel from '@/components/PixelPanel'
import type { ApiConfig, ConfigValidation, DemoStatus } from '@/utils/types'

type ConfigPanelProps = {
  config: ApiConfig
  demoStatus: DemoStatus | null
  validation: ConfigValidation | null
  onChange: (field: keyof ApiConfig, value: string) => void
  onValidate: () => void
}

const fields: Array<{ key: keyof ApiConfig; label: string; placeholder: string; masked?: boolean }> = [
  { key: 'apiKey', label: 'OPENAI_API_KEY', placeholder: 'Enter your API key', masked: true },
  { key: 'model', label: 'OPENAI_MODEL', placeholder: 'gemini-3-pro-preview' },
  { key: 'baseUrl', label: 'OPENAI_BASE_URL', placeholder: 'https://...' },
  { key: 'apiVersion', label: 'OPENAI_API_VERSION', placeholder: '2024-02-01' },
  { key: 'apiType', label: 'OPENAI_API_TYPE', placeholder: 'azure' },
]

export default function ConfigPanel({ config, demoStatus, validation, onChange, onValidate }: ConfigPanelProps) {
  return (
    <PixelPanel
      title="OpenAPI Console"
      subtitle="Use your own key, or leave the API key blank to use the server-hosted anonymous demo quota."
      action={
        <button className="pixel-btn" onClick={onValidate} type="button">
          <Radio className="h-4 w-4" />
          Validate config
        </button>
      }
    >
      <div className="space-y-4">
        {fields.map((field) => (
          <label className="block" key={field.key}>
            <span className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.25em] text-violet-100/70">
              <ShieldCheck className="h-4 w-4 text-cyan-300" />
              {field.label}
            </span>
            <input
              className="pixel-input"
              onChange={(event) => onChange(field.key, event.target.value)}
              placeholder={field.placeholder}
              type={field.masked ? 'password' : 'text'}
              value={config[field.key] ?? ''}
            />
          </label>
        ))}
      </div>
      <div className="mt-4 rounded-xl border border-violet-300/20 bg-black/20 p-4 text-sm text-violet-100/80">
        <p className="flex items-center gap-2 text-cyan-200">
          <LoaderCircle className="h-4 w-4" />
          Current mode: English profile demo, research-only workflow.
        </p>
        {demoStatus?.enabled ? (
          <p className="mt-2 text-violet-100/80">
            Anonymous demo key is available on the server. {demoStatus.remainingTurns} of {demoStatus.maxTurns} turns remain in this browser session.
          </p>
        ) : (
          <p className="mt-2 text-violet-100/70">
            No built-in demo key is configured on the server. A personal API key is required.
          </p>
        )}
        <p className="mt-2 text-violet-100/70">
          {validation ? validation.message : 'Validate the configuration before starting the chat.'}
        </p>
      </div>
    </PixelPanel>
  )
}
