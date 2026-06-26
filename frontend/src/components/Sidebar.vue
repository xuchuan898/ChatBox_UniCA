<template>
  <aside class="sidebar">
    <div>
      <div class="sidebar__title">ChatBox_UniCA</div>
      <div class="sidebar__subtitle">UniCA RAG System</div>
    </div>

    <div class="sidebar__section">
      <div class="sidebar__section-title">System Status</div>
      <StatusCard
        :online="online"
        :session-id="sessionId"
        :status="status"
        @refresh="$emit('refresh-status')"
      />
    </div>

    <div class="sidebar__section">
      <div class="sidebar__section-title">Sessions</div>
      <button class="sidebar-btn sidebar-btn--primary" @click="$emit('new-session')">
        + New Session
      </button>
      <button class="sidebar-btn sidebar-btn--danger" @click="$emit('clear-session')">
        Clear Memory
      </button>
    </div>

    <div class="sidebar__section">
      <div class="sidebar__section-title">Knowledge Base</div>
      <DocSelector
        :docs="availableDocs"
        :selected="selectedDocs"
        :loading="docsLoading"
        :building="docsBuilding"
        :error="docsError"
        @toggle="handleDocToggle"
        @apply="handleDocApply"
        @refresh="handleDocRefresh"
      />
    </div>

    <div class="sidebar__section">
      <div class="sidebar__section-title">Index</div>
      <button class="sidebar-btn" @click="$emit('rebuild-index')" :disabled="indexBuilding">
        {{ indexBuilding ? 'Building...' : 'Rebuild Index' }}
      </button>
      <div style="font-size: 12px; color: #94a3b8;">
        {{ indexBuilding ? 'Index rebuild in progress...' : '' }}
      </div>
    </div>

    <div class="sidebar__section" v-if="status.embedding_model">
      <div class="sidebar__section-title">Models</div>
      <div class="status-card">
        <div class="status-card__row">
          <span class="status-card__label">Embedding</span>
          <span class="status-card__value">{{ shortModel(status.embedding_model) }}</span>
        </div>
        <div class="status-card__row">
          <span class="status-card__label">Generation</span>
          <span class="status-card__value">{{ genModel }}</span>
        </div>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { ref } from 'vue'
import StatusCard from './StatusCard.vue'
import DocSelector from './DocSelector.vue'

const props = defineProps({
  online: { type: Boolean, default: false },
  sessionId: { type: String, default: '' },
  status: { type: Object, default: () => ({}) },
  indexBuilding: { type: Boolean, default: false },
  genModel: { type: String, default: '' },
  availableDocs: { type: Array, default: () => [] },
  selectedDocs: { type: Set, default: () => new Set() },
  docsLoading: { type: Boolean, default: false },
  docsBuilding: { type: Boolean, default: false },
  docsError: { type: String, default: '' },
})

const emit = defineEmits([
  'refresh-status', 'new-session', 'clear-session', 'rebuild-index',
  'doc-toggle', 'doc-apply', 'doc-refresh',
])

function handleDocToggle(path) {
  emit('doc-toggle', path)
}

function handleDocApply(paths) {
  emit('doc-apply', paths)
}

function handleDocRefresh() {
  emit('doc-refresh')
}

function shortModel(name) {
  if (!name) return '-'
  const parts = name.split('/')
  return parts[parts.length - 1]
}
</script>