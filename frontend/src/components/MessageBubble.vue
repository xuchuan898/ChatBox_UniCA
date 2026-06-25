<template>
  <div
    class="message"
    :class="'message--' + msg.role"
  >
    <div class="message__bubble">
      <div v-html="renderedContent"></div>
      <div v-if="msg.isStreaming" class="streaming-indicator">
        <span class="streaming-indicator__stage">
          <template v-if="currentStage === 'retrieving' || currentStage === 'retrieved'">
            Retrieving context
          </template>
          <template v-else-if="currentStage === 'generating'">
            Generating answer
          </template>
          <template v-else>
            Thinking
          </template>
        </span>
        <span class="streaming-indicator__dot"></span>
        <span class="streaming-indicator__dot"></span>
        <span class="streaming-indicator__dot"></span>
      </div>
    </div>
    <div v-if="!msg.isStreaming && msg.role === 'assistant'" class="message__meta">
      <span v-if="msg.elapsed !== undefined">{{ msg.elapsed }}s</span>
      <span v-if="msg.fromCache" class="message__cache-badge">cached</span>
    </div>
    <details v-if="!msg.isStreaming && msg.sources && msg.sources.length" class="sources-details">
      <summary>{{ msg.sources.length }} source(s)</summary>
      <div class="sources-list">
        <div v-for="(src, i) in msg.sources" :key="i" class="source-item">
          <div class="source-item__header">
            <span>{{ src.source }}</span>
            <span>{{ src.chunk_id }}</span>
          </div>
          <div class="source-item__preview">{{ src.content_preview }}</div>
        </div>
      </div>
    </details>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  msg: { type: Object, required: true },
  currentStage: { type: String, default: '' },
})

function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

function renderMarkdown(text) {
  let html = escapeHtml(text)
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  // Inline code
  html = html.replace(/`(.+?)`/g, '<code>$1</code>')
  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
  // Newlines
  html = html.replace(/\n/g, '<br/>')
  return html
}

const renderedContent = computed(() => {
  return renderMarkdown(props.msg.content)
})
</script>

<style scoped>
.message {
  display: flex;
  flex-direction: column;
  max-width: 75%;
}
.message--user {
  align-self: flex-end;
  align-items: flex-end;
}
.message--assistant {
  align-self: flex-start;
  align-items: flex-start;
}
.message__bubble {
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-wrap: break-word;
}
.message--user .message__bubble {
  background: #4f46e5;
  color: #ffffff;
  border-bottom-right-radius: 4px;
}
.message--assistant .message__bubble {
  background: #f1f5f9;
  color: #1e293b;
  border-bottom-left-radius: 4px;
}
.message__bubble code {
  background: rgba(0,0,0,0.08);
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 13px;
}
.message__meta {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  color: #64748b;
  margin-top: 4px;
  padding: 0 4px;
}
.message__cache-badge {
  background: #fef3c7;
  color: #92400e;
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 10px;
}
.sources-details {
  margin-top: 8px;
  font-size: 12px;
  width: 100%;
}
.sources-details summary {
  cursor: pointer;
  color: #4f46e5;
  font-weight: 500;
  user-select: none;
}
.sources-list {
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.source-item {
  padding: 6px 8px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  font-size: 12px;
}
.source-item__header {
  display: flex;
  justify-content: space-between;
  color: #64748b;
  margin-bottom: 2px;
}
.source-item__preview {
  color: #1e293b;
  line-height: 1.4;
}
.streaming-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #64748b;
  font-size: 12px;
  padding: 8px 0;
}
.streaming-indicator__stage {
  display: flex;
  align-items: center;
  gap: 4px;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
.streaming-indicator__dot {
  width: 6px;
  height: 6px;
  background: #4f46e5;
  border-radius: 50%;
  animation: pulse 1.2s ease-in-out infinite;
}
.streaming-indicator__dot:nth-child(2) { animation-delay: 0.2s; }
.streaming-indicator__dot:nth-child(3) { animation-delay: 0.4s; }
</style>