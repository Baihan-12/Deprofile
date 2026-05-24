import { Dice5, Layers3, Search } from 'lucide-react'

import PixelPanel from '@/components/PixelPanel'
import type { ProfileSource, ProfileSummary } from '@/utils/types'

type ProfileExplorerProps = {
  source: ProfileSource
  profiles: ProfileSummary[]
  selectedPairId: string
  filterText: string
  onSourceChange: (source: ProfileSource) => void
  onFilterChange: (value: string) => void
  onSelectProfile: (pairId: string) => void
  onRandomize: () => void
}

const sourceOptions: Array<{ value: ProfileSource; label: string; description: string }> = [
  { value: 'selected_samples', label: 'Selected samples', description: 'Browse the curated 27-profile subset' },
  { value: 'complete_index', label: 'Full index', description: 'Switch to the complete profile index with search' },
]

export default function ProfileExplorer(props: ProfileExplorerProps) {
  const filtered = props.profiles.filter((profile) => {
    const text = `${profile.pairId} ${profile.gender} ${profile.summary}`.toLowerCase()
    return text.includes(props.filterText.toLowerCase())
  })

  return (
    <PixelPanel
      title="Profile Explorer"
      subtitle="Choose the data source first, then pick a patient profile from the curated set or the full index."
      action={
        <button className="pixel-btn" onClick={props.onRandomize} type="button">
          <Dice5 className="h-4 w-4" />
          Random pick
        </button>
      }
    >
      <div className="grid gap-3 md:grid-cols-2">
        {sourceOptions.map((option) => (
          <button
            className={`source-chip ${props.source === option.value ? 'source-chip-active' : ''}`}
            key={option.value}
            onClick={() => props.onSourceChange(option.value)}
            type="button"
          >
            <span className="flex items-center gap-2 text-sm text-violet-50">
              <Layers3 className="h-4 w-4" />
              {option.label}
            </span>
            <span className="mt-2 block text-left text-xs text-violet-100/70">{option.description}</span>
          </button>
        ))}
      </div>
      <label className="mt-4 flex items-center gap-3 rounded-xl border border-white/10 bg-black/20 px-4 py-3">
        <Search className="h-4 w-4 text-cyan-200" />
        <input
          className="w-full bg-transparent text-sm text-violet-50 outline-none placeholder:text-violet-100/40"
          onChange={(event) => props.onFilterChange(event.target.value)}
          placeholder="Search by pair_id, gender, or summary keywords"
          value={props.filterText}
        />
      </label>
      <div className="mt-4 grid max-h-72 gap-3 overflow-y-auto pr-1">
        {filtered.slice(0, 120).map((profile) => (
          <button
            className={`profile-tile ${props.selectedPairId === profile.pairId ? 'profile-tile-active' : ''}`}
            key={profile.pairId}
            onClick={() => props.onSelectProfile(profile.pairId)}
            type="button"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="font-pixel text-xs text-cyan-200">#{profile.pairId}</span>
              <span className="rounded-full border border-white/10 px-2 py-1 text-[11px] text-violet-100/60">
                Candidates {profile.candidateCount}
              </span>
            </div>
            <div className="mt-3 flex flex-wrap gap-2 text-xs text-violet-50/85">
              <span>{profile.age} years</span>
              <span>{profile.gender}</span>
              <span>Depression risk {profile.depressionRiskLabel}</span>
            </div>
            <p className="mt-3 line-clamp-2 text-left text-sm text-violet-100/70">{profile.summary || 'No clinical summary available.'}</p>
          </button>
        ))}
      </div>
    </PixelPanel>
  )
}
