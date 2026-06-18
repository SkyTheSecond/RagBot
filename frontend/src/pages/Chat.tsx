import { useState, useRef, useEffect } from 'react'
import { Send, FileText, AlertTriangle, Wrench } from 'lucide-react'
import { chat, type ChatResponse, type Source } from '../lib/api'
import Markdown from '../lib/markdown'

interface Message {
  role:    'user' | 'assistant'
  content: string
  meta?:   { sources: Source[]; tool_calls_made: number; confidence_score: number }
}

export default function Chat() {
  const [messages,  setMessages]  = useState<Message[]>([])
  const [input,     setInput]     = useState('')
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | undefined>(undefined)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef  = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [input])

  async function handleSubmit() {
    const query = input.trim()
    if (!query || loading) return

    setInput('')
    if (inputRef.current) inputRef.current.style.height = '24px'
    setError(null)
    setMessages(prev => [...prev, { role: 'user', content: query }])
    setLoading(true)

    try {
      const res: ChatResponse = await chat(query, 5, sessionId)
      if (!sessionId) setSessionId(res.session_id)
      setMessages(prev => [
        ...prev,
        {
          role:    'assistant',
          content: res.answer,
          meta:    {
            sources:         res.sources,
            tool_calls_made: res.tool_calls_made,
            confidence_score:  res.confidence_score,
          },
        },
      ])
    } catch (e) {
      const raw = e instanceof Error ? e.message : 'Something went wrong'
      try {
        const parsed = JSON.parse(raw)
        setError(parsed.detail ?? parsed.message ?? raw)
      } catch {
        setError(raw)
      }
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="min-h-screen bg-paper flex flex-col">

      {/* Header */}
      <header className="border-b border-ash/40 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-dim uppercase tracking-widest">RAG</span>
          <span className="w-px h-4 bg-ash" />
          <span className="font-sans text-sm text-dim">Knowledge Chat</span>
        </div>
        <a
          href="/dashboard"
          className="font-mono text-xs text-dim hover:text-accent transition-colors"
        >
          dashboard →
        </a>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-8">
        <div className="max-w-2xl mx-auto space-y-8">

          {messages.length === 0 && (
            <div className="text-center pt-24 animate-fadein">
              <p className="font-mono text-xs text-ash uppercase tracking-widest mb-2">
                Ready
              </p>
              <p className="text-dim text-sm">Ask anything from your knowledge base.</p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className="animate-fadein">
              {msg.role === 'user' ? (
                <div className="flex justify-end">
                  <div className="bg-ink text-paper px-4 py-3 rounded-2xl rounded-br-sm max-w-md font-sans text-sm leading-relaxed">
                    {msg.content}
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  {/* Answer */}
                  <Markdown text={msg.content} />

                  {/* Meta */}
                  {msg.meta && (
                    <div className="space-y-2 pt-1">

                      {/* Low confidence warning */}
                      {msg.meta.confidence_score < 0.75 && (
                      <div className={`flex items-center gap-2 text-xs font-mono ${
                          msg.meta.confidence_score < 0.5 ? 'text-red-500' : 'text-amber-600'
                        }`}>
                          <AlertTriangle size={12} />
                          <span>
                            {msg.meta.confidence_score < 0.5
                              ? 'Answer may not be from documents'
                              : 'Partially grounded'}
                          </span>
                        </div>
                      )}

                      {/* Tool calls */}
                      {msg.meta.tool_calls_made > 0 && (
                        <div className="flex items-center gap-2 text-xs text-dim font-mono">
                          <Wrench size={12} />
                          <span>{msg.meta.tool_calls_made} tool call{msg.meta.tool_calls_made > 1 ? 's' : ''} made</span>
                        </div>
                      )}

                      {/* Sources */}
                      {msg.meta.sources.length > 0 && (
                        <div className="flex flex-wrap gap-2 pt-1">
                          {msg.meta.sources.map((s, si) => (
                            <div
                              key={si}
                              className="flex items-center gap-1.5 bg-soft border border-ash/40 rounded px-2 py-1"
                            >
                              <FileText size={10} className="text-dim flex-shrink-0" />
                              <span className="font-mono text-xs text-dim truncate max-w-[160px]">
                                {s.filename || s.doc_id}
                              </span>
                              <span className="font-mono text-xs text-ash">
                                {(s.score * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Loading indicator */}
          {loading && (
            <div className="animate-fadein">
              <div className="flex items-center gap-2 text-dim">
                <span className="font-mono text-xs">thinking</span>
                <span className="animate-blink font-mono text-accent">▋</span>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="font-mono text-xs text-red-500 animate-fadein">
              error: {error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      {/* Input */}
      <footer className="border-t border-ash/40 px-4 py-4">
        <div className="max-w-2xl mx-auto">
          <div className="flex gap-3 items-end bg-soft border border-ash/60 rounded-2xl px-4 py-2.5 focus-within:border-ink transition-colors">
            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Ask something..."
              disabled={loading}
              className="flex-1 bg-transparent resize-none outline-none font-sans text-sm text-ink placeholder-ash leading-relaxed disabled:opacity-50 overflow-y-auto"
              style={{ minHeight: '24px', height: '24px', maxHeight: '160px' }}

            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || loading}
              className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-xl bg-ink text-paper disabled:opacity-30 hover:bg-accent transition-colors mb-0.5"
            >
              <Send size={14} />
            </button>
          </div>
          <p className="text-center font-mono text-xs text-ash mt-2">
            Answers may be inaccurate. Always verify with the source document.
          </p>
        </div>
      </footer>

    </div>
  )
}
