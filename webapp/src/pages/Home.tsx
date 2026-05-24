import { Gamepad2 } from 'lucide-react'
import { useEffect, useMemo } from 'react'

import ChatPanel from '@/components/ChatPanel'
import ConfigPanel from '@/components/ConfigPanel'
import ProfileExplorer from '@/components/ProfileExplorer'
import ProfileSummaryCard from '@/components/ProfileSummaryCard'
import PromptPanel from '@/components/PromptPanel'
import { useAppStore } from '@/hooks/useAppStore'

export default function Home() {
  const {
    bootstrap,
    bootstrapping,
    config,
    demoStatus,
    detail,
    error,
    filterText,
    loadingProfile,
    messages,
    profiles,
    resetChat,
    runValidation,
    selectedPairId,
    sendMessage,
    sending,
    setConfigField,
    setFilterText,
    setSource,
    selectProfile,
    source,
    validation,
    pickRandomProfile,
  } = useAppStore()

  useEffect(() => {
    void bootstrap()
  }, [bootstrap])

  const statusText = useMemo(() => {
    if (bootstrapping) {
      return 'Scanning Deprofile assets and selected samples...'
    }
    if (error) {
      return error
    }
    if (demoStatus?.enabled && !config.apiKey) {
      return `Anonymous demo mode is available. ${demoStatus.remainingTurns} turns remain in this browser session.`
    }
    if (validation?.ok) {
      return 'Configuration is ready. You can begin chatting.'
    }
    return 'Waiting for configuration validation.'
  }, [bootstrapping, config.apiKey, demoStatus, error, validation])

  return (
    <main className="min-h-screen pixel-shell">
      <div className="scanlines" />
      <div className="pixel-grid" />
      <div className="relative mx-auto flex min-h-screen w-full max-w-[1600px] flex-col px-4 py-6 lg:px-8">
        <section className="hero-panel">
          <div>
            <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-violet-300/30 bg-violet-500/10 px-4 py-2 text-xs uppercase tracking-[0.3em] text-cyan-200">
              <Gamepad2 className="h-4 w-4" />
              DEPROFILE
            </p>
            <h1 className="hero-title">DEPROFILE Demo</h1>
            <p className="hero-subtitle">
              Pick a profile from <code>selected_samples</code>, inspect the system prompt, and chat with a simulated patient through your OpenAI-compatible endpoint.
            </p>
          </div>
          <div className="hero-status">
            <div className="status-orb" />
            <p>{statusText}</p>
            <div className="mt-4 flex flex-wrap gap-3 text-xs text-violet-50/70">
              <span className="hero-chip">Default model: gemini-3-pro-preview</span>
              <span className="hero-chip">English interface</span>
              <span className="hero-chip">Research demo only</span>
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[420px_1fr]">
          <div className="space-y-6">
            <ConfigPanel config={config} demoStatus={demoStatus} onChange={setConfigField} onValidate={() => void runValidation()} validation={validation} />
            <ProfileExplorer
              filterText={filterText}
              onFilterChange={setFilterText}
              onRandomize={() => void pickRandomProfile()}
              onSelectProfile={(pairId) => void selectProfile(pairId)}
              onSourceChange={(nextSource) => void setSource(nextSource)}
              profiles={profiles}
              selectedPairId={selectedPairId}
              source={source}
            />
          </div>

          <div className="space-y-6">
            <ProfileSummaryCard detail={detail} loading={loadingProfile || bootstrapping} />
            {detail ? <PromptPanel prompt={detail.systemPrompt} /> : null}
            <ChatPanel
                      disabled={!selectedPairId || !detail || bootstrapping || (!config.apiKey && demoStatus?.enabled === true && demoStatus.remainingTurns <= 0)}
              error={error && !bootstrapping ? error : undefined}
              messages={messages}
              onReset={resetChat}
              onSend={sendMessage}
              sending={sending}
            />
          </div>
        </section>
      </div>
    </main>
  )
}
