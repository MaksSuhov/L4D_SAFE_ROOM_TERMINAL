/**
 * Left 4 Dead survivor accents used by the avatar tiles and the message
 * labels. The backend stores only names, so the identity of a character in the
 * UI is derived from the name itself; the palette keeps the well-known
 * survivors visually distinct and falls back to a stable hash for anyone the
 * archive gains later.
 */
const SURVIVOR_ACCENTS = {
  Билл: '#c2703a',
  Зои: '#d9528f',
  Луис: '#4aa8d8',
  Фрэнсис: '#8d7bd0',
  Эллис: '#e0b23c',
  Ник: '#4fb98b',
  Рошель: '#d2604f',
  Тренер: '#7fb03c',
}

const FALLBACK_ACCENTS = ['#e0b23c', '#4fb98b', '#d2604f', '#5aa9e6', '#8d7bd0', '#d9528f']

/** Stable, order-independent hash so an unknown name always gets one colour. */
function hashName(name) {
  let hash = 5381
  for (const character of name) {
    hash = ((hash << 5) + hash + character.codePointAt(0)) >>> 0
  }
  return hash
}

/**
 * Resolve the accent of a survivor into ready-to-use CSS values: the base
 * colour plus translucent variants for borders, glows and backgrounds.
 */
export function personaAccent(name = '') {
  const accent =
    SURVIVOR_ACCENTS[name] ?? FALLBACK_ACCENTS[hashName(name) % FALLBACK_ACCENTS.length]

  return {
    accent,
    border: `${accent}80`,
    glow: `${accent}59`,
    background: `${accent}1f`,
  }
}

/** Two-letter monogram for the avatar tile, e.g. `Эллис` -> `ЭЛ`. */
export function personaInitials(name = '') {
  const words = name.trim().split(/\s+/).filter(Boolean)
  if (words.length === 0) return '??'
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase()
  return words
    .slice(0, 2)
    .map((word) => word[0])
    .join('')
    .toUpperCase()
}
