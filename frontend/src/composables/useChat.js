import { ref, reactive, markRaw } from 'vue'
import { sendMessage, sendMessageStream } from '../api/chat.js'
import { createSession, clearSession as apiClearSession } from '../api/sessions.js'

export function useChat() {
  const messages = reactive([])
  const sessionId = ref(null)
  const streaming = ref(false)
  const generating = ref(false)
  const retrieving = ref(false)
  const loading = ref(false)
  const error = ref(null)
  const streamAnswer = ref('')
  const currentStage = ref('')

  // SSE streaming control
  let abortController = null

  async function initSession() {
    try {
      const data = await createSession()
      sessionId.value = data.session_id
    } catch (e) {
      error.value = 'Failed to create session: ' + e.message
    }
  }

  function addMessage(role, content, extras = {}) {
    messages.push({
      role,
      content,
      id: Date.now() + Math.random(),
      ...extras,
    })
  }

  async function send(query, useStream) {
    if (!query.trim()) return

    if (!sessionId.value) {
      await initSession()
    }

    error.value = null
    addMessage('user', query)

    if (useStream) {
      await _sendStream(query)
    } else {
      await _sendJson(query)
    }
  }

  async function _sendJson(query) {
    loading.value = true
    try {
      const data = await sendMessage(query, sessionId.value)
      addMessage('assistant', data.answer, {
        sources: data.sources || [],
        elapsed: data.elapsed_seconds,
        fromCache: data.from_cache,
      })
    } catch (e) {
      error.value = 'Request failed: ' + e.message
    } finally {
      loading.value = false
    }
  }

  async function _sendStream(query) {
    streaming.value = true
    retrieving.value = true
    currentStage.value = 'retrieving'
    streamAnswer.value = ''

    addMessage('assistant', '', {
      isStreaming: true,
      sources: [],
      elapsed: 0,
      fromCache: false,
    })
    const msgIndex = messages.length - 1

    abortController = new AbortController()

    try {
      await sendMessageStream(
        query,
        sessionId.value,
        {},
        (event) => {
          const { stage, ...data } = event
          if (stage === 'retrieving') {
            retrieving.value = true
            generating.value = false
            currentStage.value = 'retrieving'
          } else if (stage === 'retrieved') {
            retrieving.value = false
            currentStage.value = 'retrieved'
          } else if (stage === 'generating') {
            generating.value = true
            currentStage.value = 'generating'
          } else if (stage === 'done') {
            streaming.value = false
            generating.value = false
            retrieving.value = false
            currentStage.value = 'done'
            // Update the message with final data
            messages[msgIndex] = {
              role: 'assistant',
              content: data.answer || streamAnswer.value,
              id: messages[msgIndex].id,
              sources: data.sources || [],
              elapsed: data.elapsed_seconds,
              fromCache: data.from_cache,
            }
          } else if (stage === 'session') {
            if (data.session_id) {
              sessionId.value = data.session_id
            }
          } else if (stage === 'cache_check' || stage === 'error') {
            // no-op
          }
        }
      )
    } catch (e) {
      if (e.name !== 'AbortError') {
        error.value = 'Stream failed: ' + e.message
      }
      streaming.value = false
      generating.value = false
      retrieving.value = false
      currentStage.value = ''
    }
  }

  async function newSession() {
    try {
      const data = await createSession()
      sessionId.value = data.session_id
      messages.splice(0, messages.length)
      error.value = null
    } catch (e) {
      error.value = 'Failed to create session: ' + e.message
    }
  }

  async function clearCurrentSession() {
    if (!sessionId.value) return
    try {
      await apiClearSession(sessionId.value)
      messages.splice(0, messages.length)
      await initSession()
    } catch (e) {
      error.value = 'Failed to clear session: ' + e.message
    }
  }

  async function switchToSession(sid) {
    sessionId.value = sid
    messages.splice(0, messages.length)
    error.value = null
  }

  function clearMessages() {
    messages.splice(0, messages.length)
  }

  return {
    messages,
    sessionId,
    streaming,
    loading,
    error,
    streamAnswer,
    currentStage,
    retrieving,
    generating,
    send,
    newSession,
    clearCurrentSession,
    clearMessages,
    initSession,
    switchToSession,
  }
}