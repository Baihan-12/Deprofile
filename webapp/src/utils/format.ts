export function prettifySymptom(symptom: string): string {
  const parts = symptom.split('-')
  return parts[parts.length - 1]?.trim() || symptom
}

export function bigFiveEntries(values: Record<string, number>): Array<{ label: string; value: number }> {
  return Object.entries(values).map(([label, value]) => ({ label, value }))
}

export function riskTone(level: string): string {
  if (level === 'High') {
    return 'text-[#f3e9ff] border-[#9B8EC7]/75 bg-[#9B8EC7]/24'
  }
  if (level === 'Moderate' || level === 'Mild') {
    return 'text-[#eefcff] border-[#B4D3D9]/80 bg-[#B4D3D9]/22'
  }
  return 'text-[#f6f0ff] border-[#BDA6CE]/75 bg-[#BDA6CE]/20'
}
