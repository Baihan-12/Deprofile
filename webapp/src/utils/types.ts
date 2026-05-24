export type ProfileSource = 'selected_samples' | 'complete_index'

export type ApiConfig = {
  apiKey: string
  model: string
  baseUrl: string
  apiVersion?: string
  apiType?: string
}

export type ProfileSummary = {
  pairId: string
  age: number | string
  gender: string
  maritalStatus: string
  workStatus: string
  depressionRisk: number | string
  depressionRiskLabel: string
  suicideRisk: number | string
  suicideRiskLabel: string
  candidateCount: number
  crId: string
  d4Id: string
  bigFive: Record<string, number>
  positiveSymptoms: string[]
  negativeSymptoms: string[]
  summary: string
}

export type ProfileDetail = {
  profile: ProfileSummary
  timelinePreview: string[]
  systemPrompt: string
}

export type ChatTurn = {
  role: 'user' | 'assistant'
  content: string
}

export type ConfigValidation = {
  ok: boolean
  message: string
}

export type DemoStatus = {
  enabled: boolean
  maxTurns: number
  usedTurns: number
  remainingTurns: number
}

export type ChatResponse = {
  reply: string
  systemPrompt: string
  profile: ProfileSummary
  timelinePreview: string[]
  demoStatus: DemoStatus
  usedDemoKey: boolean
}
