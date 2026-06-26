<template>
  <div class="app-layout">
    <Sidebar
      :online="online"
      :session-id="sessionId"
      :status="indexStatus"
      :index-building="indexBuilding"
      :gen-model="genModel"
      @refresh-status="refreshStatus"
      @new-session="handleNewSession"
      @clear-session="handleClearSession"
      @rebuild-index="handleRebuildIndex"
    />

    <ChatInterface
      :messages="messages"
      :loading="loading"
      :streaming="streaming"
      :error="error"
      :current-stage="currentStage"
      :use-stream="useStream"
      @send="handleSend"
      @toggle-stream="useStream = !useStream"
    />

    <ToastContainer />
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, markRaw } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatInterface from './components/ChatInterface.vue'
import ToastContainer from './components/ToastContainer.vue'
import { useChat } from './composables/useChat.js'
import { useToast } from './composables/useToast.js'
import { getIndexStatus, rebuildIndex as apiRebuildIndex } from './api/index.js'
import { get } from './api/client.js'

const {
  messages,
  sessionId,
  streaming,
  loading,
  error,
  currentStage,
  send,
  newSession,
  clearCurrentSession,
  initSession,
} = useChat()

const { info, success, error: toastError } = useToast()

const useStream = ref(true)
const online = ref(false)
const indexStatus = reactive({})
const indexBuilding = ref(false)
const genModel = ref('')

async function refreshStatus() {
  try {
    const health = await get('/health')
    online.value = health.status === 'ok'
    const idx = await getIndexStatus()
    Object.assign(indexStatus, idx)
    info('Status refreshed')
  } catch (e) {
    online.value = false
    toastError('Failed to fetch status: ' + e.message)
  }
}

async function handleSend(query) {
  try {
    await send(query, useStream.value)
  } catch (e) {
    toastError(e.message)
  }
}

async function handleNewSession() {
  await newSession()
  success('New session created')
}

async function handleClearSession() {
  await clearCurrentSession()
  info('Session memory cleared')
}

async function handleRebuildIndex() {
  try {
    indexBuilding.value = true
    await apiRebuildIndex()
    success('Index rebuild started')
    // Poll status until done
    const poll = setInterval(async () => {
      try {
        const idx = await getIndexStatus()
        Object.assign(indexStatus, idx)
        if (idx.is_ready) {
          clearInterval(poll)
          indexBuilding.value = false
          success('Index rebuild completed')
        }
      } catch {
        // still building
      }
    }, 3000)
  } catch (e) {
    indexBuilding.value = false
    toastError('Rebuild failed: ' + e.message)
  }
}

onMounted(async () => {
  await initSession()
  await refreshStatus()
  // Try to read generation model from config endpoint
  try {
    const cfg = await get('/api/v1/config/')
    if (cfg.generation && cfg.generation.model_name) {
      genModel.value = cfg.generation.model_name
    }
  } catch {
    // ignore
  }
})
</script>