import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import ProfileSummaryCard from '@/components/ProfileSummaryCard'
import type { ProfileDetail } from '@/utils/types'

const detail: ProfileDetail = {
  profile: {
    pairId: '0069',
    age: 36,
    gender: 'F',
    maritalStatus: 'married',
    workStatus: 'Unknown',
    depressionRisk: 2,
    depressionRiskLabel: 'Moderate',
    suicideRisk: 1,
    suicideRiskLabel: 'Mild',
    candidateCount: 3,
    crId: '2',
    d4Id: '948',
    bigFive: {
      Openness: 2,
      Neuroticism: 6,
    },
    positiveSymptoms: ['PHQ-insomnia'],
    negativeSymptoms: ['PHQ-hypersomnia'],
    summary: 'Test summary',
  },
  timelinePreview: ['T-4d | event | tweet'],
  systemPrompt: 'prompt',
}

describe('ProfileSummaryCard', () => {
  it('renders profile summary content', () => {
    render(<ProfileSummaryCard detail={detail} loading={false} />)
    expect(screen.getByText('Patient profile')).toBeInTheDocument()
    expect(screen.getByText('Test summary')).toBeInTheDocument()
    expect(screen.getByText('insomnia')).toBeInTheDocument()
  })
})
