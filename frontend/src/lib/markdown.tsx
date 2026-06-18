import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'

const components: Components = {
  p:  ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
  h1: ({ children }) => <h2 className="font-mono text-sm text-dim uppercase tracking-widest mt-4 mb-2">{children}</h2>,
  h2: ({ children }) => <h3 className="font-mono text-xs text-dim uppercase tracking-widest mt-3 mb-2">{children}</h3>,
  h3: ({ children }) => <h4 className="font-mono text-xs text-dim uppercase tracking-widest mt-3 mb-1">{children}</h4>,
  strong: ({ children }) => <strong className="font-medium text-ink">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  ul: ({ children }) => <ul className="list-disc pl-5 space-y-1 mb-2">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal pl-5 space-y-1 mb-2">{children}</ol>,
  li: ({ children }) => <li>{children}</li>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-ash pl-3 text-dim italic mb-2">{children}</blockquote>
  ),
  hr: () => <hr className="border-ash/40 my-3" />,
  table: ({ children }) => (
    <div className="overflow-x-auto mb-2">
      <table className="border border-ash/40 rounded-lg text-sm w-full">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-soft">{children}</thead>,
  th: ({ children }) => (
    <th className="border border-ash/40 px-2 py-1 text-left font-mono text-xs text-dim uppercase">
      {children}
    </th>
  ),
  td: ({ children }) => <td className="border border-ash/40 px-2 py-1">{children}</td>,
}

export default function Markdown({ text }: { text: string }) {
  return (
    <div className="font-sans text-sm leading-relaxed text-ink">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  )
}
