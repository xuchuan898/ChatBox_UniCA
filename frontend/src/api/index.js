import { get, post as httpPost } from './client'

export function getIndexStatus() {
  return get('/api/v1/index/status')
}

export function rebuildIndex() {
  return httpPost('/api/v1/index/rebuild')
}