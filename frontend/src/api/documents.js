import { get, post } from './client'

export function listDocuments() {
  return get('/api/v1/documents/list')
}

export function getActiveDocuments() {
  return get('/api/v1/documents/active')
}

export function selectDocuments(selected) {
  return post('/api/v1/documents/select', { selected })
}