import PixelPanel from '@/components/PixelPanel'
import { bigFiveEntries, prettifySymptom, riskTone } from '@/utils/format'
import type { ProfileDetail } from '@/utils/types'

type ProfileSummaryCardProps = {
  detail: ProfileDetail | null
  loading: boolean
}

export default function ProfileSummaryCard({ detail, loading }: ProfileSummaryCardProps) {
  if (loading) {
    return (
      <PixelPanel title="Patient profile" className="animate-pulse" subtitle="Loading profile evidence and timeline">
        Loading...
      </PixelPanel>
    )
  }

  if (!detail) {
    return <PixelPanel title="Patient profile">Choose a profile to inspect its evidence bundle.</PixelPanel>
  }

  const { profile, timelinePreview } = detail

  return (
    <PixelPanel title="Patient profile" subtitle="An English-facing summary built from the selected Deprofile record.">
      <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <InfoStat label="Age" value={`${profile.age}`} />
            <InfoStat label="Gender" value={profile.gender} />
            <InfoStat label="Marital" value={profile.maritalStatus} />
            <InfoStat label="Work" value={profile.workStatus} />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
              <p className="pixel-label">Risk labels</p>
              <div className="mt-3 flex flex-wrap gap-3 text-sm">
                <span className={`rounded-full border px-3 py-1 ${riskTone(profile.depressionRiskLabel)}`}>
                  Depression {profile.depressionRiskLabel}
                </span>
                <span className={`rounded-full border px-3 py-1 ${riskTone(profile.suicideRiskLabel)}`}>
                  Suicide {profile.suicideRiskLabel}
                </span>
              </div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
              <p className="pixel-label">Big Five</p>
              <div className="mt-3 grid gap-2 text-sm text-violet-50/80">
                {bigFiveEntries(profile.bigFive).map((item) => (
                  <div className="grid grid-cols-[110px_1fr_30px] items-center gap-3" key={item.label}>
                    <span>{item.label}</span>
                    <div className="h-2 rounded-full bg-white/10">
                      <div className="h-full rounded-full bg-gradient-to-r from-cyan-300 to-violet-400" style={{ width: `${Math.min(item.value * 16, 100)}%` }} />
                    </div>
                    <span className="text-right text-cyan-200">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <TagList title="Positive symptoms" items={profile.positiveSymptoms.map(prettifySymptom).slice(0, 12)} />
            <TagList title="Negative symptoms" items={profile.negativeSymptoms.map(prettifySymptom).slice(0, 12)} />
          </div>
          <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
            <p className="pixel-label">Clinical summary</p>
            <p className="mt-3 text-sm leading-7 text-violet-50/80">{profile.summary || 'No clinical summary available.'}</p>
          </div>
        </div>
        <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
          <p className="pixel-label">Life-event timeline</p>
          <div className="mt-4 space-y-3">
            {timelinePreview.length ? (
              timelinePreview.map((item) => (
                <div className="rounded-xl border border-violet-300/20 bg-violet-500/5 p-3 text-sm text-violet-50/80" key={item}>
                  {item}
                </div>
              ))
            ) : (
              <p className="text-sm text-violet-100/60">No life-event timeline was found for the top candidate.</p>
            )}
          </div>
        </div>
      </div>
    </PixelPanel>
  )
}

function InfoStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
      <p className="text-xs uppercase tracking-[0.25em] text-violet-100/50">{label}</p>
      <p className="mt-3 font-pixel text-xs text-cyan-200">{value}</p>
    </div>
  )
}

function TagList({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
      <p className="pixel-label">{title}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {items.map((item) => (
          <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-violet-50/80" key={item}>
            {item}
          </span>
        ))}
      </div>
    </div>
  )
}
