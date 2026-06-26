<template>
  <details class="override-panel">
    <summary>Parameter Overrides</summary>
    <div class="override-panel__body">
      <div class="override-field">
        <span class="override-field__label">
          <span>rerank_candidates</span>
          <span>{{ local.rerank_candidates }}</span>
        </span>
        <input
          type="range"
          class="override-field__input"
          min="5"
          max="50"
          step="1"
          :value="local.rerank_candidates"
          @input="local.rerank_candidates = Number($event.target.value)"
        />
      </div>
      <div class="override-field">
        <span class="override-field__label">
          <span>dynamic_topk_ratio</span>
          <span>{{ local.dynamic_topk_ratio }}</span>
        </span>
        <input
          type="range"
          class="override-field__input"
          min="0"
          max="1"
          step="0.05"
          :value="local.dynamic_topk_ratio"
          @input="local.dynamic_topk_ratio = Number($event.target.value)"
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
          @input="local.weight_vec = Number($event.target.value)"
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
          @input="local.weight_bm25 = Number($event.target.value)"
        />
      </div>
      <div class="override-field">
        <span class="override-field__label">
          <span>temperature</span>
          <span>{{ local.temperature }}</span>
        </span>
        <input
          type="range"
          class="override-field__input"
          min="0"
          max="1"
          step="0.05"
          :value="local.temperature"
          @input="local.temperature = Number($event.target.value)"
        />
      </div>
      <div class="override-field">
        <span class="override-field__label">
          <span>num_predict</span>
          <span>{{ local.num_predict }}</span>
        </span>
        <input
          type="range"
          class="override-field__input"
          min="64"
          max="4096"
          step="32"
          :value="local.num_predict"
          @input="local.num_predict = Number($event.target.value)"
        />
      </div>
      <div class="override-field">
        <label class="override-field__checkbox">
          <input
            type="checkbox"
            :checked="local.query_expansion !== false"
            @change="local.query_expansion = $event.target.checked ? null : false"
          />
          Enable query expansion
        </label>
      </div>
      <div class="override-panel__actions">
        <button class="override-btn override-btn--apply" @click="applyAndNotify">Apply</button>
        <button class="override-btn override-btn--save" @click="saveAndNotify">Apply & Save to Disk</button>
      </div>
    </div>
  </details>
</template>

<script setup>
import { reactive } from 'vue'
import { updateConfig, writeConfig } from '../api/config.js'
import { useToast } from '../composables/useToast.js'

const { success, error: showError } = useToast()

const defaults = {
  rerank_candidates: 30,
  dynamic_topk_ratio: 0.5,
  weight_vec: 1.25,
  weight_bm25: 0.75,
  temperature: 0.0,
  num_predict: 256,
  query_expansion: null,
}

const local = reactive({ ...defaults })

function buildOverrides() {
  const o = {}
  if (local.rerank_candidates !== 30) o['retrieval.rerank_candidates'] = local.rerank_candidates
  if (local.dynamic_topk_ratio !== 0.5) o['retrieval.dynamic_topk_ratio'] = local.dynamic_topk_ratio
  if (local.weight_vec !== 1.25) o['retrieval.weight_vec'] = local.weight_vec
  if (local.weight_bm25 !== 0.75) o['retrieval.weight_bm25'] = local.weight_bm25
  if (local.temperature !== 0.0) o['generation.temperature'] = local.temperature
  if (local.num_predict !== 256) o['generation.num_predict'] = local.num_predict
  if (local.query_expansion === false) o['query_expansion.enabled'] = false
  return o
}

async function applyAndNotify() {
  const overrides = buildOverrides()
  if (!Object.keys(overrides).length) {
    success('No changes to apply')
    return
  }
  try {
    await updateConfig(overrides)
    success('Config applied')
  } catch (e) {
    showError('Apply failed: ' + e.message)
  }
}

async function saveAndNotify() {
  const overrides = buildOverrides()
  if (!Object.keys(overrides).length) {
    success('No changes to save')
    return
  }
  try {
    await writeConfig(overrides)
    success('Config saved to disk')
  } catch (e) {
    showError('Save failed: ' + e.message)
  }
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
.override-panel__actions {
  display: flex;
  gap: 8px;
  margin-top: 4px;
}
.override-btn {
  flex: 1;
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s;
}
.override-btn--apply {
  background: #4f46e5;
  color: #fff;
  border-color: #4f46e5;
}
.override-btn--apply:hover {
  background: #4338ca;
}
.override-btn--save {
  background: #fff;
  color: #1e293b;
}
.override-btn--save:hover {
  background: #f1f5f9;
}
</style>