<template>
  <div class="status-card">
    <div class="status-card__row">
      <span class="status-card__label">Status</span>
      <span class="status-card__badge">
        <span
          class="status-card__dot"
          :class="online ? 'status-card__dot--online' : 'status-card__dot--offline'"
        ></span>
        {{ online ? 'Online' : 'Offline' }}
      </span>
    </div>
    <div class="status-card__row">
      <span class="status-card__label">Session</span>
      <span class="status-card__value" :title="sessionId">{{ sessionId ? sessionId.slice(0, 8) + '...' : '-' }}</span>
    </div>
    <div class="status-card__row" v-if="status.doc_count !== undefined">
      <span class="status-card__label">Documents</span>
      <span class="status-card__value">{{ status.doc_count }}</span>
    </div>
    <div class="status-card__row" v-if="status.vector_count !== undefined">
      <span class="status-card__label">Chunks</span>
      <span class="status-card__value">{{ status.vector_count }}</span>
    </div>
    <button class="sidebar-btn sidebar-btn--primary" style="margin-top: 4px;" @click="$emit('refresh')">
      Refresh Status
    </button>
  </div>
</template>

<script setup>
defineProps({
  online: { type: Boolean, default: false },
  sessionId: { type: String, default: '' },
  status: { type: Object, default: () => ({}) },
})

defineEmits(['refresh'])
</script>