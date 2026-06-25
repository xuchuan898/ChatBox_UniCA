<template>
  <details class="override-panel">
    <summary>Parameter Overrides</summary>
    <div class="override-panel__body">
      <div class="override-field">
        <span class="override-field__label">
          <span>top_k</span>
          <span>{{ local.top_k }}</span>
        </span>
        <input
          type="range"
          class="override-field__input"
          min="1"
          max="20"
          step="1"
          :value="local.top_k"
          @input="update('top_k', Number($event.target.value))"
        />
      </div>
      <div class="override-field">
        <span class="override-field__label">
          <span>rerank_alpha</span>
          <span>{{ local.rerank_alpha }}</span>
        </span>
        <input
          type="range"
          class="override-field__input"
          min="0"
          max="1"
          step="0.05"
          :value="local.rerank_alpha"
          @input="update('rerank_alpha', Number($event.target.value))"
        />
      </div>
      <div class="override-field">
        <span class="override-field__label">
          <span>weight_vec</span>
          <span>{{ local.weight_vec }}</span>
        </span>
        <input
          type="range"
          class="override-field__input"
          min="0"
          max="2"
          step="0.05"
          :value="local.weight_vec"
          @input="update('weight_vec', Number($event.target.value))"
        />
      </div>
      <div class="override-field">
        <span class="override-field__label">
          <span>weight_bm25</span>
          <span>{{ local.weight_bm25 }}</span>
        </span>
        <input
          type="range"
          class="override-field__input"
          min="0"
          max="2"
          step="0.05"
          :value="local.weight_bm25"
          @input="update('weight_bm25', Number($event.target.value))"
        />
      </div>
      <div class="override-field">
        <label class="override-field__checkbox">
          <input
            type="checkbox"
            :checked="local.query_expansion !== false"
            @change="update('query_expansion', $event.target.checked ? null : false)"
          />
          Enable query expansion
        </label>
      </div>
    </div>
  </details>
</template>

<script setup>
import { reactive, watch } from 'vue'

const props = defineProps({
  modelValue: { type: Object, default: () => ({}) },
})

const emit = defineEmits(['update:modelValue'])

const defaults = {
  top_k: 5,
  rerank_alpha: 0.5,
  weight_vec: 1.0,
  weight_bm25: 1.0,
  query_expansion: null,
}

const local = reactive({ ...defaults, ...props.modelValue })

watch(() => props.modelValue, (val) => {
  Object.assign(local, { ...defaults, ...val })
})

function update(key, value) {
  local[key] = value
  const payload = { ...local }
  // Build override dict
  const overrides = {}
  if (payload.top_k !== 5) overrides.top_k = payload.top_k
  if (payload.rerank_alpha !== 0.5) overrides['retrieval.rerank_alpha'] = payload.rerank_alpha
  if (payload.weight_vec !== 1.0) overrides['retrieval.weight_vec'] = payload.weight_vec
  if (payload.weight_bm25 !== 1.0) overrides['retrieval.weight_bm25'] = payload.weight_bm25
  if (payload.query_expansion === false) overrides['query_expansion.enabled'] = false
  emit('update:modelValue', overrides)
}
</script>

<style scoped>
.override-panel {
  margin-top: 8px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
}
.override-panel summary {
  padding: 8px 12px;
  font-size: 12px;
  cursor: pointer;
  color: #64748b;
  user-select: none;
  background: #f0f2f5;
}
.override-panel summary:hover {
  background: #e2e8f0;
}
.override-panel__body {
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.override-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.override-field__label {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #64748b;
}
.override-field__input {
  width: 100%;
  accent-color: #4f46e5;
}
.override-field__checkbox {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: #1e293b;
  cursor: pointer;
}
</style>