import { useState, useEffect, useRef } from 'react'
import {
  Upload, Trash2, RefreshCw, FileText, File, FileCode,
  CheckCircle2, XCircle, Loader2, AlertTriangle,
} from 'lucide-react'
import {
  listDocuments, uploadDocument, deleteDocument,
  reingestDocument, getHealth,
  type Document, type HealthResponse,
} from '../lib/api'

type Status = 'idle' | 'loading' | 'success' | 'error'

function FileIcon({ type }: { type: string }) {
  if (type === 'pdf') return <FileText size={14} className="text-accent" />
  if (type === 'md')  return <FileCode  size={14} className="text-blue-500" />
  return <File size={14} className="text-dim" />
}

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span className={`inline-block w-1.5 h-1.5 rounded-full ${ok ? 'bg-green-500' : 'bg-red-400'}`} />
  )
}

export default function Dashboard() {
  const [docs,    setDocs]    = useState<Document[]>([])
  const [health,  setHealth]  = useState<HealthResponse | null>(null)
  const [status,  setStatus]  = useState<Status>('idle')
  const [message, setMessage] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  // Per-row action state
  const [rowAction, setRowAction] = useState<Record<string, 'deleting' | 'reingesting' | null>>({})

  const fileInputRef      = useRef<HTMLInputElement>(null)
  const reingestInputRef  = useRef<HTMLInputElement>(null)
  const reingestTarget    = useRef<string | null>(null)

  useEffect(() => { load() }, [])

  async function load() {
    setStatus('loading')
    try {
      const [d, h] = await Promise.all([listDocuments(), getHealth()])
      setDocs(d)
      setHealth(h)
      setStatus('idle')
    } catch (e) {
      setStatus('error')
      setMessage(e instanceof Error ? e.message : 'Failed to load')
    }
  }

  async function handleUpload(file: File) {
    setStatus('loading')
    setMessage(null)
    try {
      const doc = await uploadDocument(file)
      setDocs(prev => [...prev, doc])
      setMessage(`✓ ${doc.filename} ingested`)
      setStatus('success')
    } catch (e) {
      const raw = e instanceof Error ? e.message : 'Upload failed'
      try { setMessage(JSON.parse(raw).detail ?? raw) } catch { setMessage(raw) }
      setStatus('error')
    }
  }

  async function handleDelete(doc_id: string) {
    setRowAction(prev => ({ ...prev, [doc_id]: 'deleting' }))
    try {
      await deleteDocument(doc_id)
      setDocs(prev => prev.filter(d => d.id !== doc_id))
    } catch (e) {
      const raw = e instanceof Error ? e.message : 'Delete failed'
      try { setMessage(JSON.parse(raw).detail ?? raw) } catch { setMessage(raw) }

    } finally {
      setRowAction(prev => ({ ...prev, [doc_id]: null }))
    }
  }

  async function handleReingest(doc_id: string, file: File) {
    setRowAction(prev => ({ ...prev, [doc_id]: 'reingesting' }))
    try {
      const updated = await reingestDocument(doc_id, file)
      setDocs(prev => prev.map(d => d.id === doc_id ? updated : d))
      setMessage(`✓ ${updated.filename} reingested`)
    } catch (e) {
      const raw = e instanceof Error ? e.message : 'Reingest failed'
      try { setMessage(JSON.parse(raw).detail ?? raw) } catch { setMessage(raw) }

    } finally {
      setRowAction(prev => ({ ...prev, [doc_id]: null }))
    }
  }

  function onDropzoneFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) handleUpload(file)
    e.target.value = ''
  }

  function onReingestFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file && reingestTarget.current) handleReingest(reingestTarget.current, file)
    e.target.value = ''
    reingestTarget.current = null
  }

  function triggerReingest(doc_id: string) {
    reingestTarget.current = doc_id
    reingestInputRef.current?.click()
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleUpload(file)
  }

  return (
    <div className="min-h-screen bg-paper">

      {/* Hidden file inputs */}
      <input ref={fileInputRef}     type="file" accept=".pdf,.md,.txt" className="hidden" onChange={onDropzoneFile} />
      <input ref={reingestInputRef} type="file" accept=".pdf,.md,.txt" className="hidden" onChange={onReingestFile} />

      {/* Header */}
      <header className="border-b border-ash/40 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-dim uppercase tracking-widest">RAG</span>
          <span className="w-px h-4 bg-ash" />
          <span className="font-sans text-sm text-dim">Dashboard</span>
        </div>
        <a href="/chat" className="font-mono text-xs text-dim hover:text-accent transition-colors">
          chat →
        </a>
      </header>

      <main className="max-w-3xl mx-auto px-4 py-10 space-y-8">

        {/* Health */}
        <div className="flex items-center gap-6 font-mono text-xs text-dim min-h-[20px]">
        {health ? (
          <>

            <span className="flex items-center gap-1.5">
              <StatusDot ok={health.mongo_ok} /> mongo
            </span>
            <span className="flex items-center gap-1.5">
              <StatusDot ok={health.qdrant_ok} /> qdrant
            </span>
            <span className="flex items-center gap-1.5">
              <StatusDot ok={health.embedding_ok} /> embedding
            </span>
            <button onClick={load} className="ml-auto hover:text-ink transition-colors">
              <RefreshCw size={11} />
            </button>
          </>
        ) : (
          <span className="text-ash">checking status...</span>
        )}
      </div>

        {/* Upload dropzone */}
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            border-2 border-dashed rounded-2xl px-6 py-10 text-center cursor-pointer
            transition-all duration-200
            ${dragOver
              ? 'border-accent bg-accent/5'
              : 'border-ash/60 hover:border-ink/40 hover:bg-soft/60'}
          `}
        >
          <Upload size={20} className="mx-auto mb-3 text-dim" />
          <p className="font-sans text-sm text-dim">
            Drop a file or <span className="text-ink font-medium">click to upload</span>
          </p>
          <p className="font-mono text-xs text-ash mt-1">.pdf · .md · .txt</p>
        </div>

        {/* Status message */}
        {message && (
          <div className={`flex items-center gap-2 font-mono text-xs animate-fadein ${
            status === 'error' ? 'text-red-500' : 'text-green-600'
          }`}>
            {status === 'error'
              ? <XCircle size={12} />
              : status === 'success'
                ? <CheckCircle2 size={12} />
                : <AlertTriangle size={12} />
            }
            {message}
          </div>
        )}

        {/* Document list */}
        <div className="space-y-2">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-mono text-xs text-dim uppercase tracking-widest">
              Documents
            </h2>
            <span className="font-mono text-xs text-ash">{docs.length}</span>
          </div>

          {status === 'loading' && docs.length === 0 && (
            <div className="flex items-center gap-2 text-dim font-mono text-xs py-4">
              <Loader2 size={12} className="animate-spin" />
              loading...
            </div>
          )}

          {docs.length === 0 && status !== 'loading' && (
            <p className="font-mono text-xs text-ash py-4">
              No documents ingested yet.
            </p>
          )}

          {docs.map(doc => (
            <div
              key={doc.id}
              className="flex items-center gap-3 bg-soft border border-ash/40 rounded-xl px-4 py-3 animate-fadein group"
            >
              <FileIcon type={doc.filetype} />

              <div className="flex-1 min-w-0">
                <p className="font-sans text-sm text-ink truncate">{doc.filename}</p>
                <p className="font-mono text-xs text-ash">
                  {doc.filetype} · {(doc.char_count / 1000).toFixed(1)}k chars
                </p>
              </div>

              <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">

                {/* Reingest */}
                <button
                  onClick={() => triggerReingest(doc.id)}
                  disabled={!!rowAction[doc.id]}
                  title="Reingest"
                  className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-ash/30 text-dim hover:text-ink transition-colors disabled:opacity-40"
                >
                  {rowAction[doc.id] === 'reingesting'
                    ? <Loader2 size={13} className="animate-spin" />
                    : <RefreshCw size={13} />
                  }
                </button>

                {/* Delete */}
                <button
                  onClick={() => handleDelete(doc.id)}
                  disabled={!!rowAction[doc.id]}
                  title="Delete"
                  className="w-7 h-7 flex items-center justify-center rounded-lg hover:bg-red-50 text-dim hover:text-red-500 transition-colors disabled:opacity-40"
                >
                  {rowAction[doc.id] === 'deleting'
                    ? <Loader2 size={13} className="animate-spin" />
                    : <Trash2 size={13} />
                  }
                </button>

              </div>
            </div>
          ))}
        </div>

      </main>
    </div>
  )
}
