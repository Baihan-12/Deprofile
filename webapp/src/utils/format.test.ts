import { describe, expect, it } from 'vitest'

import { bigFiveEntries, prettifySymptom, riskTone } from '@/utils/format'

describe('format utils', () => {
  it('extracts symptom suffix', () => {
    expect(prettifySymptom('PHQ-insomnia')).toBe('insomnia')
  })

  it('turns big five into ordered entries', () => {
    expect(bigFiveEntries({ Openness: 2, Neuroticism: 6 })).toHaveLength(2)
  })

  it('maps high risk into the custom palette', () => {
    expect(riskTone('High')).toContain('#9B8EC7')
  })
})
