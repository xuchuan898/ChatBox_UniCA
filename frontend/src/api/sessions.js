import { post, del } from './client'

export function createSession() {
  return post('/api/v1/sessions/new')
}

export function clearSession(sessionId) {
  return del(`/api/v1/sessions/${sessionId}`)
}