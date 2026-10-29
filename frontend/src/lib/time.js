/** Wall-clock stamp printed next to every transmission. */
export function formatClock(timestamp) {
  if (!timestamp) return ''
  return new Date(timestamp).toLocaleTimeString('ru-RU', {
    hour: '2-digit',
    minute: '2-digit',
  })
}
