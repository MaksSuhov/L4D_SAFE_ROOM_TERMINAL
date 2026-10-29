/**
 * Presentation helpers for the lore archive.
 *
 * The backend answers with a tier (`source_type`) that says how much a record
 * can be trusted — verified lore, a rumour from a safe-room wall, or a recorded
 * line of the survivor who is speaking — plus the place the record came from.
 * The terminal prints both the way a survivor would.
 */
const SOURCE_TYPE_LABELS = {
  official: 'ОФИЦИАЛЬНЫЙ ЛОР',
  graffiti: 'ГРАФФИТИ НА СТЕНЕ',
  voice: 'РЕЧЬ ПЕРСОНАЖА',
}

/** Accents distinguishing the tiers, so a rumour never reads like a fact. */
const SOURCE_TYPE_ACCENTS = {
  official: 'text-term-green/80 border-term-green/25',
  graffiti: 'text-term-amber/90 border-term-amber/25',
  voice: 'text-zinc-400 border-zinc-600/50',
}

const FALLBACK_ACCENT = 'text-zinc-400 border-zinc-600/50'

/** Human-readable label for a record's evidence tier. */
export function sourceTypeLabel(sourceType) {
  if (!sourceType) return 'АРХИВ'
  return SOURCE_TYPE_LABELS[sourceType.toLowerCase()] ?? sourceType.toUpperCase()
}

/** Tailwind classes marking the tier of a record. */
export function sourceTypeAccent(sourceType) {
  if (!sourceType) return FALLBACK_ACCENT
  return SOURCE_TYPE_ACCENTS[sourceType.toLowerCase()] ?? FALLBACK_ACCENT
}

/** Clamp a cosine similarity into the 0..1 range used by the progress bars. */
export function similarityRatio(similarity) {
  if (typeof similarity !== 'number' || Number.isNaN(similarity)) return 0
  return Math.min(Math.max(similarity, 0), 1)
}

/** Render a cosine similarity as a percentage. */
export function formatSimilarity(similarity) {
  return `${Math.round(similarityRatio(similarity) * 100)}%`
}
