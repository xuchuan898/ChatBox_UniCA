import { get, post, del } from './client'

export function listSessions() {
  return get('/api/v1/sessions/list')
}

export function createSession() {
  return post('/api/v1/sessions/new')
}

export function clearSession(sessionId) {
  return del(`/api/v1/sessions/${sessionId}`)
}

export function clearAllSessions() {
  return del('/api/v1/sessions/')
}