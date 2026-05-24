import type { ApiConfig, ChatResponse, ChatTurn, ConfigValidation, DemoStatus, ProfileDetail, ProfileSource, ProfileSummary } from '@/utils/types'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function withApiBase(path: string): string {
  return `${API_BASE_URL}${path}`
}

async function requestJson<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const message = payload?.detail ?? 'Request failed'
    throw new Error(message)
  }

  return response.json() as Promise<T>
}

export async function fetchDefaults(): Promise<ApiConfig> {
  const payload = await requestJson<{ config: ApiConfig }>(withApiBase('/api/config/defaults'))
  return payload.config
}

export async function validateConfig(config: ApiConfig, sessionId: string): Promise<ConfigValidation> {
  return requestJson<ConfigValidation>(withApiBase('/api/config/validate'), {
    method: 'POST',
    body: JSON.stringify({ config, sessionId }),
  })
}

export async function fetchDemoStatus(sessionId: string): Promise<DemoStatus> {
  const query = new URLSearchParams({ session_id: sessionId }).toString()
  return requestJson<DemoStatus>(withApiBase(`/api/demo/status?${query}`))
}

export async function fetchProfiles(source: ProfileSource): Promise<ProfileSummary[]> {
  const query = new URLSearchParams({ source }).toString()
  const payload = await requestJson<{ items: ProfileSummary[] }>(withApiBase(`/api/profiles?${query}`))
  return payload.items
}

export async function fetchProfileDetail(source: ProfileSource, pairId: string): Promise<ProfileDetail> {
  const query = new URLSearchParams({ source, language: 'en' }).toString()
  return requestJson<ProfileDetail>(withApiBase(`/api/profiles/${pairId}?${query}`))
}

export async function sendChatMessage(params: {
  config: ApiConfig
  source: ProfileSource
  pairId: string
  messages: ChatTurn[]
  sessionId: string
}): Promise<ChatResponse> {
  return requestJson<ChatResponse>(withApiBase('/api/chat'), {
    method: 'POST',
    body: JSON.stringify({
      ...params,
      language: 'en',
    }),
  })
}
