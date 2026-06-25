const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

async function request(method, path, body = null) {
  const url = BASE_URL + path
  const options = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body !== null) {
    options.body = JSON.stringify(body)
  }
  const res = await fetch(url, options)
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(errBody.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export function get(path) {
  return request('GET', path)
}

export function post(path, body) {
  return request('POST', path, body)
}

export function del(path) {
  return request('DELETE', path)
}

export function put(path, body) {
  return request('PUT', path, body)
}

export async function postStream(path, body, onEvent) {
  const url = BASE_URL + path
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(errBody.detail || `HTTP ${res.status}`)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (trimmed.startsWith('data: ')) {
        try {
          const data = JSON.parse(trimmed.slice(6))
          onEvent(data)
        } catch {
          // skip malformed events
        }
      }
    }
  }
}