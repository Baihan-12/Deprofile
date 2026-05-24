import { create } from 'zustand'

import { fetchDefaults, fetchDemoStatus, fetchProfileDetail, fetchProfiles, sendChatMessage, validateConfig } from '@/utils/api'
import type { ApiConfig, ChatTurn, ConfigValidation, DemoStatus, ProfileDetail, ProfileSource, ProfileSummary } from '@/utils/types'

type AppState = {
  config: ApiConfig
  sessionId: string
  demoStatus: DemoStatus | null
  source: ProfileSource
  profiles: ProfileSummary[]
  selectedPairId: string
  detail: ProfileDetail | null
  messages: ChatTurn[]
  validation: ConfigValidation | null
  filterText: string
  bootstrapping: boolean
  loadingProfile: boolean
  sending: boolean
  error: string
  setConfigField: (field: keyof ApiConfig, value: string) => void
  setSource: (source: ProfileSource) => Promise<void>
  setFilterText: (value: string) => void
  bootstrap: () => Promise<void>
  selectProfile: (pairId: string) => Promise<void>
  pickRandomProfile: () => Promise<void>
  runValidation: () => Promise<void>
  resetChat: () => void
  sendMessage: (content: string) => Promise<void>
}

const SESSION_STORAGE_KEY = 'deprofile-demo-session-id'

function getOrCreateSessionId(): string {
  if (typeof window === 'undefined') {
    return 'server-session'
  }
  const existing = window.localStorage.getItem(SESSION_STORAGE_KEY)
  if (existing) {
    return existing
  }
  const next = window.crypto?.randomUUID?.() ?? `session-${Date.now()}`
  window.localStorage.setItem(SESSION_STORAGE_KEY, next)
  return next
}

const emptyConfig: ApiConfig = {
  apiKey: '',
  model: 'gemini-3-pro-preview',
  baseUrl: 'https://aidp.bytedance.net/api/modelhub/online/v2/crawl',
  apiVersion: '2024-02-01',
  apiType: 'azure',
}

export const useAppStore = create<AppState>((set, get) => ({
  config: emptyConfig,
  sessionId: '',
  demoStatus: null,
  source: 'selected_samples',
  profiles: [],
  selectedPairId: '',
  detail: null,
  messages: [],
  validation: null,
  filterText: '',
  bootstrapping: true,
  loadingProfile: false,
  sending: false,
  error: '',
  setConfigField: (field, value) => {
    set((state) => ({
      config: {
        ...state.config,
        [field]: value,
      },
    }))
  },
  setFilterText: (value) => set({ filterText: value }),
  bootstrap: async () => {
    set({ bootstrapping: true, error: '' })
    try {
      const sessionId = getOrCreateSessionId()
      const [config, demoStatus, profiles] = await Promise.all([
        fetchDefaults(),
        fetchDemoStatus(sessionId),
        fetchProfiles('selected_samples'),
      ])
      const selectedPairId = profiles[0]?.pairId ?? ''
      let detail: ProfileDetail | null = null
      if (selectedPairId) {
        detail = await fetchProfileDetail('selected_samples', selectedPairId)
      }
      set({ config, sessionId, demoStatus, profiles, selectedPairId, detail, messages: [], bootstrapping: false })
    } catch (error) {
      set({ bootstrapping: false, error: error instanceof Error ? error.message : 'Failed to initialize the demo' })
    }
  },
  setSource: async (source) => {
    set({ source, loadingProfile: true, error: '', filterText: '' })
    try {
      const profiles = await fetchProfiles(source)
      const selectedPairId = profiles[0]?.pairId ?? ''
      let detail: ProfileDetail | null = null
      if (selectedPairId) {
        detail = await fetchProfileDetail(source, selectedPairId)
      }
      set({ profiles, selectedPairId, detail, loadingProfile: false, messages: [] })
    } catch (error) {
      set({ loadingProfile: false, error: error instanceof Error ? error.message : 'Failed to switch profile source' })
    }
  },
  selectProfile: async (pairId) => {
    const source = get().source
    set({ selectedPairId: pairId, loadingProfile: true, error: '' })
    try {
      const detail = await fetchProfileDetail(source, pairId)
      set({ detail, loadingProfile: false, messages: [] })
    } catch (error) {
      set({ loadingProfile: false, error: error instanceof Error ? error.message : 'Failed to load profile details' })
    }
  },
  pickRandomProfile: async () => {
    const profiles = get().profiles
    if (!profiles.length) {
      return
    }
    const randomItem = profiles[Math.floor(Math.random() * profiles.length)]
    await get().selectProfile(randomItem.pairId)
  },
  runValidation: async () => {
    try {
      const { config, sessionId } = get()
      const validation = await validateConfig(config, sessionId)
      set({ validation, error: '' })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Failed to validate the configuration' })
    }
  },
  resetChat: () => set({ messages: [], error: '' }),
  sendMessage: async (content) => {
    const trimmed = content.trim()
    if (!trimmed) {
      return
    }
    const { config, source, selectedPairId, messages, sessionId } = get()
    const nextMessages: ChatTurn[] = [...messages, { role: 'user', content: trimmed }]
    set({ messages: nextMessages, sending: true, error: '' })
    try {
      const response = await sendChatMessage({ config, source, pairId: selectedPairId, messages: nextMessages, sessionId })
      set({
        messages: [...nextMessages, { role: 'assistant', content: response.reply }],
        detail: { profile: response.profile, timelinePreview: response.timelinePreview, systemPrompt: response.systemPrompt },
        demoStatus: response.demoStatus,
        sending: false,
        error: '',
      })
    } catch (error) {
      set({
        messages: nextMessages,
        sending: false,
        error: error instanceof Error ? error.message : 'Failed to send the message',
      })
    }
  },
}))
