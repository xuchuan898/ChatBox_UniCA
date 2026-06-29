<template>
  <div class="doc-selector">
    <div class="doc-selector__label">
      <span>Documents</span>
      <button
        v-if="!loading"
        class="doc-selector__refresh"
        @click="$emit('refresh')"
        title="Refresh document list"
      >&#x21bb;</button>
    </div>

    <div v-if="loading" class="doc-selector__loading">Loading documents...</div>

    <div v-else-if="error" class="doc-selector__error">{{ error }}</div>

    <div v-else-if="!docs.length" class="doc-selector__empty">
      No documents found in docs/chroma/
    </div>

    <div v-else class="doc-selector__list">
      <label
        v-for="doc in docs"
        :key="doc.name"
        class="doc-selector__item"
        :class="{ 'doc-selector__item--active': selected.has(doc.path) }"
      >
        <input
          type="checkbox"
          :checked="selected.has(doc.path)"
          :disabled="building"
          @change="toggle(doc.path)"
        />
        <span class="doc-selector__item-name">{{ doc.name }}</span>
        <span class="doc-selector__item-size">{{ (doc.size_bytes / 1024).toFixed(0) }}KB</span>
      </label>
    </div>

    <div class="doc-selector__actions">
      <button
        class="doc-selector__apply-btn"
        :disabled="building || !selected.size"
        @click="$emit('apply', [...selected])"
      >
        <span v-if="building">Building index...</span>
        <span v-else>Apply ({{ selected.size }} selected)</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  docs: { type: Array, default: () => [] },
  selected: { type: Set, default: () => new Set() },
  loading: { type: Boolean, default: false },
  building: { type: Boolean, default: false },
  error: { type: String, default: '' },
})

const emit = defineEmits(['toggle', 'apply', 'refresh'])

function toggle(path) {
  emit('toggle', path)
}
</script>

<style scoped>
.doc-selector {
  margin-top: 4px;
  padding: 0 4px;
}
.doc-selector__label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 11px;
  font-weight: 600;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
.doc-selector__refresh {
  background: none;
  border: none;
  color: #64748b;
  cursor: pointer;
  font-size: 14px;
  padding: 0;
  line-height: 1;
}
.doc-selector__refresh:hover {
  color: #4f46e5;
}
.doc-selector__loading,
.doc-selector__empty,
.doc-selector__error {
  font-size: 11px;
  color: #94a3b8;
  padding: 4px 0;
}
.doc-selector__error {
  color: #ef4444;
}
.doc-selector__list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 180px;
  overflow-y: auto;
}
.doc-selector__item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 4px;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  color: #64748b;
  transition: background 0.1s;
}
.doc-selector__item:hover {
  background: #1e293b;
}
.doc-selector__item--active {
  color: #e2e8f0;
}
.doc-selector__item input[type="checkbox"] {
  accent-color: #4f46e5;
  margin: 0;
}
.doc-selector__item-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.doc-selector__item-size {
  font-size: 10px;
  color: #475569;
  flex-shrink: 0;
}
.doc-selector__actions {
  margin-top: 6px;
}
.doc-selector__apply-btn {
  width: 100%;
  padding: 5px 8px;
  font-size: 11px;
  border-radius: 4px;
  border: 1px solid #334155;
  background: #1e293b;
  color: #94a3b8;
  cursor: pointer;
  transition: all 0.15s;
}
.doc-selector__apply-btn:hover:not(:disabled) {
  background: #4f46e5;
  border-color: #4f46e5;
  color: #fff;
}
.doc-selector__apply-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
</style>