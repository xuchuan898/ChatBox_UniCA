import { put, post } from './client'

export async function updateConfig(overrides) {
  return put('/api/v1/config/', { overrides })
}

export async function writeConfig(overrides) {
  return post('/api/v1/config/write', { overrides })
}