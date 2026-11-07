/**
 * The double-submit token: the server writes it into a cookie that page scripts
 * are allowed to read, and the client must repeat it in a request header. A
 * foreign site can make the browser send the session cookie, but it cannot read
 * this one to put the same value into the header.
 */
export function readCsrfToken(cookieName) {
  if (!cookieName || typeof document === 'undefined' || !document.cookie) return ''

  const prefix = `${cookieName}=`
  const entry = document.cookie
    .split(';')
    .map((cookie) => cookie.trim())
    .find((cookie) => cookie.startsWith(prefix))

  return entry ? decodeURIComponent(entry.slice(prefix.length)) : ''
}
