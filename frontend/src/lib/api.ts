const BASE = '/v1'

export interface Document {
  id:         string
  filename:   string
  filetype:   string
  char_count: number
  filepath:   string
}

export interface Source {
  chunk_id: string
  doc_id:   string
  score:    number
  filename: string
}

export interface ChatResponse {
  answer:          string
  confidence_score:  number
  tool_calls_made: number
  sources:         Source[]
  session_id:      string
}

export interface HealthResponse {
  mongo_ok:     boolean
  qdrant_ok:    boolean
  embedding_ok: boolean
  details:      Record<string, unknown>
}

export interface DeleteResponse {
  doc_id:  string
  deleted: Record<string, number>
}

// ── Chat ────────────────────────────────────────────────────────────────────

export async function chat(query: string, top_k = 5, session_id?: string): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ query, top_k, session_id }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function* chatStream(
  query: string,
  top_k = 5,
  session_id: string,
): AsyncGenerator<string> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ query, top_k, session_id }),
  })
  if (!res.ok) throw new Error(await res.text())

  const reader  = res.body!.getReader()
  const decoder = new TextDecoder()
  let   buffer  = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('data: ')) yield line.slice(6)
    }
  }
}

// ── Documents ────────────────────────────────────────────────────────────────

export async function listDocuments(): Promise<Document[]> {
  const res = await fetch(`${BASE}/documents`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function uploadDocument(file: File): Promise<Document> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/documents/upload`, {
    method: 'POST',
    body:   form,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function deleteDocument(doc_id: string): Promise<DeleteResponse> {
  const res = await fetch(`${BASE}/documents/${doc_id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function reingestDocument(
  doc_id: string,
  file:   File,
): Promise<Document> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/documents/${doc_id}/reingest`, {
    method: 'POST',
    body:   form,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
