<template>
  <details class="debug-panel">
    <summary>
      <span class="debug-panel__title">Debug Retrieval</span>
    </summary>
    <div class="debug-panel__body">
      <div class="debug-panel__input-row">
        <input
          ref="inputRef"
          v-model="query"
          class="debug-panel__input"
          placeholder="Enter a test query..."
          @keydown.enter.prevent="handleQuery"
        />
        <button
          class="debug-panel__submit"
          :disabled="!query.trim() || loading"
          @click="handleQuery"
        >Go</button>
      </div>

      <div v-if="loading" class="debug-panel__status">Querying...</div>
      <div v-else-if="err" class="debug-panel__error">{{ err }}</div>

      <div v-else-if="result" class="debug-panel__result">
        <div class="debug-panel__result-meta">
          <strong>{{ result.final_doc_count }}</strong> documents retrieved
        </div>
        <div class="debug-panel__result-list">
          <div
            v-for="item in result.reranked"
            :key="item.rank"
            class="debug-panel__item"
            :class="{ 'debug-panel__item--low': item.score < 0.3 }"
          >
            <div class="debug-panel__item-header">
              <span class="debug-panel__item-rank">#{{ item.rank }}</span>
              <span class="debug-panel__item-score">{{ item.score.toFixed(4) }}</span>
              <span class="debug-panel__item-source">{{ item.source }}</span>
            </div>
            <div class="debug-panel__item-preview">{{ item.content_preview }}</div>
          </div>
        </div>
      </div>

      <div v-else class="debug-panel__hint">Enter a query and click Go to test retrieval</div>
    </div>
  </details>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { get } from '../api/client.js'

const query = ref('')
const loading = ref(false)
const result = ref(null)
const err = ref('')
const inputRef = ref(null)

async function handleQuery() {
  const q = query.value.trim()
  if (!q || loading.value) return
  loading.value = true
  err.value = ''
  result.value = null
  try {
    const data = await get('/api/v1/debug/retrieve?query=' + encodeURIComponent(q))
    if (data.error) {
      err.value = data.error
    } else {
      result.value = data
    }
  } catch (e) {
    err.value = e.message
  } finally {
    loading.value = false
  }
}

nextTick(() => {
  if (inputRef.value) inputRef.value.focus()
})
</script>

<style scoped>
.debug-panel {
  margin-top: 0;
  border: 1px solid #334155;
  border-radius: 6px;
  overflow: hidden;
}
.debug-panel[open] {
  background: #0f172a;
}
.debug-panel summary {
  padding: 6px 8px;
  font-size: 11px;
  cursor: pointer;
  color: #94a3b8;
  user-select: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.debug-panel summary:hover {
  color: #e2e8f0;
}
.debug-panel__title {
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.debug-panel__body {
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  border-top: 1px solid #334155;
}
.debug-panel__input-row {
  display: flex;
  gap: 4px;
}
.debug-panel__input {
  flex: 1;
  padding: 4px 6px;
  font-size: 11px;
  border: 1px solid #334155;
  border-radius: 4px;
  background: #1e293b;
  color: #e2e8f0;
  outline: none;
}
.debug-panel__input:focus {
  border-color: #4f46e5;
}
.debug-panel__submit {
  padding: 4px 10px;
  font-size: 11px;
  border-radius: 4px;
  border: 1px solid #4f46e5;
  background: #4f46e5;
  color: #fff;
  cursor: pointer;
}
.debug-panel__submit:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.debug-panel__submit:hover:not(:disabled) {
  background: #4338ca;
}
.debug-panel__status {
  font-size: 11px;
  color: #94a3b8;
}
.debug-panel__error {
  font-size: 11px;
  color: #ef4444;
}
.debug-panel__hint {
  font-size: 10px;
  color: #64748b;
  font-style: italic;
}
.debug-panel__result-meta {
  font-size: 11px;
  color: #94a3b8;
}
.debug-panel__result-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 300px;
  overflow-y: auto;
}
.debug-panel__item {
  padding: 5px 6px;
  border-radius: 4px;
  background: #1e293b;
  border: 1px solid #1e293b;
}
.debug-panel__item--low {
  opacity: 0.6;
}
.debug-panel__item-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 2px;
}
.debug-panel__item-rank {
  font-size: 10px;
  font-weight: 600;
  color: #64748b;
  min-width: 18px;
}
.debug-panel__item-score {
  font-size: 10px;
  font-family: monospace;
  color: #4f46e5;
  font-weight: 600;
}
.debug-panel__item-source {
  font-size: 10px;
  color: #64748b;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.debug-panel__item-preview {
  font-size: 10px;
  color: #94a3b8;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>