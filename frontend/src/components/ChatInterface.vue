<template>
  <div class="chat-area">
    <!-- Header -->
    <div class="chat-header">
      <div>
        <div class="chat-header__title">ChatBox_UniCA</div>
        <div class="chat-header__subtitle">RAG Q&A for Universit&eacute; C&ocirc;te d'Azur CS Master</div>
      </div>
      <div class="chat-header__controls">
        <label class="toggle" @click="$emit('toggle-stream')">
          <span>JSON</span>
          <div class="toggle__switch" :class="{ 'toggle__switch--active': useStream }">
            <div class="toggle__knob"></div>
          </div>
          <span>SSE</span>
        </label>
      </div>
    </div>

    <!-- Error banner -->
    <div v-if="error" class="error-banner">{{ error }}</div>

    <!-- Loading indicator for non-streaming -->
    <div v-if="loading" class="streaming-indicator" style="padding: 16px 24px;">
      <span class="streaming-indicator__dot"></span>
      <span class="streaming-indicator__dot"></span>
      <span class="streaming-indicator__dot"></span>
      <span style="margin-left: 4px;">Processing...</span>
    </div>

    <!-- Messages -->
    <div class="messages" ref="messagesRef">
      <template v-if="messages.length">
        <MessageBubble
          v-for="msg in messages"
          :key="msg.id"
          :msg="msg"
          :currentStage="currentStage"
        />
      </template>
      <div v-else class="empty-chat">
        <div class="empty-chat__icon">💬</div>
        <div class="empty-chat__text">Ask a question</div>
        <div class="empty-chat__hint">Type your question below and press Enter to start</div>
      </div>
    </div>

    <!-- Input -->
    <div class="input-area">
      <div class="input-area__row">
        <textarea
          ref="textareaRef"
          class="input-area__textarea"
          v-model="inputText"
          :disabled="loading || streaming"
          placeholder="Ask a question..."
          rows="1"
          @keydown.enter.prevent="onSend"
          @input="autoResize"
        ></textarea>
        <button
          class="input-area__send-btn"
          :disabled="!inputText.trim() || loading || streaming"
          @click="onSend"
        >
          &#10148;
        </button>
      </div>
      <div class="input-area__hint">Press Enter to send, Shift+Enter for new line</div>

      <OverrideControls />
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue'
import MessageBubble from './MessageBubble.vue'
import OverrideControls from './OverrideControls.vue'

const props = defineProps({
  messages: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
  streaming: { type: Boolean, default: false },
  error: { type: String, default: '' },
  currentStage: { type: String, default: '' },
  useStream: { type: Boolean, default: true },
})

const emit = defineEmits(['send', 'toggle-stream'])

const inputText = ref('')
const messagesRef = ref(null)
const textareaRef = ref(null)

function autoResize() {
  const el = textareaRef.value
  if (el) {
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }
}

function onSend() {
  const text = inputText.value.trim()
  if (!text || props.loading || props.streaming) return
  inputText.value = ''
  if (textareaRef.value) {
    textareaRef.value.style.height = 'auto'
  }
  emit('send', text)
}

// Auto-scroll on new messages
watch(
  () => props.messages.length,
  async () => {
    await nextTick()
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  }
)

// Focus textarea on mount
nextTick(() => {
  if (textareaRef.value) textareaRef.value.focus()
})
</script>