<template>
  <div class="app-layout">
    <Sidebar
      :online="online"
      :session-id="sessionId"
      :status="indexStatus"
      :index-building="indexBuilding"
      :gen-model="genModel"
      :available-docs="availableDocs"
      :selected-docs="selectedDocs"
      :docs-loading="docsLoading"
      :docs-building="docsBuilding"
      :docs-error="docsError"
      @refresh-status="refreshStatus"
      @new-session="handleNewSession"
      @clear-session="handleClearSession"
      @rebuild-index="handleRebuildIndex"
      @doc-toggle="handleDocToggle"
      @doc-apply="handleDocApply"
      @doc-refresh="handleDocRefresh"
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
import { ref, reactive, onMounted } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatInterface from './components/ChatInterface.vue'
import ToastContainer from './components/ToastContainer.vue'
import { useChat } from './composables/useChat.js'
import { useToast } from './composables/useToast.js'
import { getIndexStatus, rebuildIndex as apiRebuildIndex } from './api/index.js'
import { listDocuments, selectDocuments } from './api/documents.js'
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

// Document selection state
const availableDocs = ref([])
const selectedDocs = ref(new Set())
const docsLoading = ref(false)
const docsBuilding = ref(false)
const docsError = ref('')

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

// Document selection handlers
async function handleDocRefresh() {
  docsLoading.value = true
  docsError.value = ''
  try {
    const docs = await listDocuments()
    availableDocs.value = docs
    // Load currently active selection
    const active = await (await fetch('/api/v1/documents/active')).json()
    if (active.selected && active.selected.length) {
      selectedDocs.value = new Set(active.selected)
    }
  } catch (e) {
    docsError.value = e.message
  } finally {
    docsLoading.value = false
  }
}

function handleDocToggle(path) {
  const next = new Set(selectedDocs.value)
  if (next.has(path)) {
    next.delete(path)
  } else {
    next.add(path)
  }
  selectedDocs.value = next
}

async function handleDocApply(paths) {
  if (!paths.length) return
  docsBuilding.value = true
  docsError.value = ''
  try {
    const result = await selectDocuments(paths)
    success(`Index built: ${result.doc_count} docs, ${result.chunk_count} chunks`)
    // Update indexStatus with new counts
    indexStatus.doc_count = result.doc_count
    indexStatus.vector_count = result.chunk_count
    indexStatus.is_ready = true
  } catch (e) {
    docsError.value = e.message
    toastError('Doc selection failed: ' + e.message)
  } finally {
    docsBuilding.value = false
  }
}

onMounted(async () => {
  await initSession()
  await refreshStatus()
  // Load document list
  await handleDocRefresh()
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