import { post, postStream } from './client'

export async function sendMessage(query, sessionId, overrides = {}) {
  return post('/api/v1/chat/', { query, session_id: sessionId, overrides })
}

export async function sendMessageStream(query, sessionId, overrides = {}, onEvent) {
  return postStream('/api/v1/chat/stream', { query, session_id: sessionId, overrides }, onEvent)
}