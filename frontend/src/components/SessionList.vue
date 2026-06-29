<template>
  <div class="session-list">
    <div class="session-list__actions">
      <button class="sidebar-btn sidebar-btn--primary" @click="$emit('new-session')">
        + New
      </button>
      <button class="sidebar-btn sidebar-btn--danger" @click="handleClearAll" :disabled="clearLoading">
        Clear All
      </button>
    </div>

    <div v-if="loading" class="session-list__loading">Loading sessions...</div>
    <div v-else-if="error" class="session-list__error">{{ error }}</div>
    <div v-else-if="!sessions.length" class="session-list__empty">No sessions</div>

    <div v-else class="session-list__items">
      <div
        v-for="s in sessions"
        :key="s.session_id"
        class="session-list__item"
        :class="{ 'session-list__item--active': s.session_id === activeSessionId }"
        @click="switchTo(s.session_id)"
      >
        <div class="session-list__item-top">
          <span class="session-list__item-id">{{ shortId(s.session_id) }}</span>
          <span v-if="s.session_id === activeSessionId" class="session-list__item-badge">active</span>
        </div>
        <div class="session-list__item-meta">
          {{ s.turn_count || 0 }} msgs
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { listSessions, createSession, clearAllSessions } from '../api/sessions.js'
import { useToast } from '../composables/useToast.js'

const props = defineProps({
  activeSessionId: { type: String, default: '' },
})

const emit = defineEmits(['new-session', 'switch-session', 'sessions-cleared'])

const { success, error: showError } = useToast()

const sessions = ref([])
const loading = ref(false)
const error = ref('')
const clearLoading = ref(false)

async function fetchSessions() {
  loading.value = true
  error.value = ''
  try {
    sessions.value = await listSessions()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

async function switchTo(sessionId) {
  emit('switch-session', sessionId)
}

async function handleClearAll() {
  clearLoading.value = true
  try {
    await clearAllSessions()
    sessions.value = []
    success('All sessions cleared')
    emit('sessions-cleared')
  } catch (e) {
    showError('Failed to clear sessions: ' + e.message)
  } finally {
    clearLoading.value = false
  }
}

function shortId(id) {
  if (!id) return '-'
  return id.length > 12 ? id.slice(0, 12) + '...' : id
}

// Watch for sessionId changes to refresh the list
import { watch } from 'vue'
watch(() => props.activeSessionId, () => {
  fetchSessions()
})

defineExpose({ fetchSessions })

onMounted(fetchSessions)
</script>

<style scoped>
.session-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.session-list__actions {
  display: flex;
  gap: 4px;
}
.session-list__actions .sidebar-btn {
  flex: 1;
  padding: 4px 8px;
  font-size: 11px;
  border-radius: 4px;
  border: 1px solid #334155;
  background: #1e293b;
  color: #94a3b8;
  cursor: pointer;
  transition: all 0.15s;
}
.session-list__actions .sidebar-btn--primary:hover {
  background: #4f46e5;
  border-color: #4f46e5;
  color: #fff;
}
.session-list__actions .sidebar-btn--danger:hover {
  background: #dc2626;
  border-color: #dc2626;
  color: #fff;
}
.session-list__actions .sidebar-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.session-list__loading,
.session-list__empty,
.session-list__error {
  font-size: 11px;
  color: #94a3b8;
  padding: 4px 0;
}
.session-list__error {
  color: #ef4444;
}
.session-list__items {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 140px;
  overflow-y: auto;
}
.session-list__item {
  padding: 5px 6px;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.1s;
}
.session-list__item:hover {
  background: #1e293b;
}
.session-list__item--active {
  background: #1e293b;
  border-left: 2px solid #4f46e5;
}
.session-list__item-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.session-list__item-id {
  font-size: 11px;
  color: #e2e8f0;
  font-family: monospace;
}
.session-list__item-badge {
  font-size: 9px;
  background: #4f46e5;
  color: #fff;
  padding: 1px 5px;
  border-radius: 3px;
  text-transform: uppercase;
}
.session-list__item-meta {
  font-size: 10px;
  color: #64748b;
  margin-top: 1px;
}
</style>