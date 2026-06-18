# %% Imports
from __future__ import annotations

import json
from typing import Any, Literal

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from inference_client import InferenceClient, ChatMessage, Tool, ToolFunction
from qdrant_store import QdrantStore, QueryResult
from mongo_store import MongoStore
# %% State

class RAGState(BaseModel):
    query: str
    history: list[dict[str, str]] = []
    effective_query: str = ""
    retrieved_chunks: list[QueryResult] = []
    messages: list[dict[str, Any]] = []
    tool_call_count: int = 0
    answer: str = ""
    confidence_score: float = 1.0

    class Config:
        arbitrary_types_allowed = True


# %% Tool definitions (Ollama format)

MAX_ITERATIONS = 3

MONGO_TOOLS = [
    Tool(
        function=ToolFunction(
            name="get_surrounding_chunks",
            description=(
                "Fetch the N chunks before and after a given chunk in the document. "
                "Use when a retrieved chunk lacks enough surrounding context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "The chunk ID to expand around"},
                    "n":        {"type": "integer", "description": "Number of neighbors in each direction", "default": 2},
                },
                "required": ["chunk_id"],
            },
        )
    ),
    Tool(
        function=ToolFunction(
            name="get_section_chunks",
            description=(
              "Fetch all chunks belonging to a heading section. "
              "Use this when the question is about a specific topic or section by name, "
              "or when surrounding chunks aren't enough and you need everything under a heading. "
              "Prefer this over get_full_document — it is cheaper and usually sufficient."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "section_id": {"type": "string", "description": "The heading section ID"},
                },
                "required": ["section_id"],
            },
        )
    ),
    Tool(
        function=ToolFunction(
            name="get_full_document",
            description=(
                "Fetch the complete text of a document. "
                "Use only as a last resort when chunk-level context is insufficient."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The document ID"},
                },
                "required": ["doc_id"],
            },
        )
    ),
]


# %% Tool executor

def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    mongo: MongoStore,
) -> str:
    """Execute a tool call and return a string result for the LLM."""
    try:
        if name == "get_surrounding_chunks":
            chunks = mongo.get_surrounding_chunks(
                chunk_id=arguments["chunk_id"],
                n=int(arguments.get("n", 2)),
            )
            return json.dumps([
                {"chunk_id": c.id, "content": c.content, "position": c.position}
                for c in chunks
            ])

        elif name == "get_section_chunks":
            chunks = mongo.get_semantic_chunks_for_section(
                section_id=arguments["section_id"]
            )
            heading = mongo.get_heading_chunk(arguments["section_id"])
            return json.dumps({
                "heading": heading.heading_text if heading else "Unknown",
                "chunks": [
                    {"chunk_id": c.id, "content": c.content, "position": c.position}
                    for c in chunks
                ],
            })

        elif name == "get_full_document":
            print(f"[tool] get_full_document called with: {arguments}")
            doc = mongo.get_document(arguments["doc_id"])
            print(f"[tool] get_full_document result: {doc is not None}")
            if doc is None:
                return json.dumps({"error": f"Document {arguments['doc_id']} not found"})
            
            content = doc.content[:8000]
            truncated = len(doc.content) > 8000
            result = json.dumps({
                "filename": doc.filename,
                "content": content,
                "truncated": truncated,
                "total_chars": len(doc.content)
            })
            print(f"[tool] get_full_document returning {result} chars")
            return result


        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


#%% Truncate tool
def _trim_tool_results(messages: list[dict], max_chars: int = 6000) -> list[dict]:
    """Truncate oversized tool result content so the context window doesn't explode."""
    trimmed = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if len(content) > max_chars:
                msg = {**msg, "content": content[:max_chars] + "\n...[truncated]"}
        trimmed.append(msg)
    return trimmed

# %% RAG Graph

class RAGPipeline:
    """
    LangGraph-based RAG pipeline.

    Flow:
        retrieve → generate ──(tool call?)──→ execute_tool → generate (loop, max 3)
                           └──(answer)──────→ grade → END
    """

    def __init__(
        self,
        client: InferenceClient,
        qdrant: QdrantStore,
        mongo: MongoStore,
        top_k: int = 5,
    ) -> None:
        self._client = client
        self._qdrant = qdrant
        self._mongo  = mongo
        self._top_k  = top_k
        self._graph  = self._build_graph()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def _condense_history(self, older: list[dict[str, str]]) -> str:
        """Summarize older conversation turns into a compact note."""
        summary_messages = [
            ChatMessage(
                role="system",
                content=(
                    "Summarize the following conversation history into a short, dense "
                    "paragraph capturing the key facts, topics, and conclusions established "
                    "so far. Do not include pleasantries or restate the question-answer "
                    "format — just the substantive content someone would need to follow "
                    "the conversation. Output only the summary, nothing else."
                ),
            ),
            ChatMessage(
                role="user",
                content="\n".join(f"{m['role']}: {m['content']}" for m in older),
            ),
        ]
        response = self._client.chat(messages=summary_messages)
        return response.content or ""
  
    def _build_history_context(self, history: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Keep the last 4 messages verbatim; condense anything older into one summary note."""
        if len(history) <= 4:
            return history
  
        older, recent = history[:-4], history[-4:]
        summary = self._condense_history(older)
        return [
            {"role": "system", "content": f"Summary of earlier conversation: {summary}"},
            *recent,
        ]

    def _node_retrieve(self, state: RAGState) -> dict[str, Any]:
        """Embed the query and fetch top_k chunks from Qdrant."""
        query = state.query
        if state.history:
            rewrite_messages = [
                ChatMessage(
                    role="system",
                    content=(
                        "You are a search query rewriter.\n"
                        "Given a conversation history and a follow-up question, rewrite the follow-up "
                        "into a fully self-contained search query that can be understood with no prior context.\n"
                        "Rules:\n"
                        "1. Identify the core factual conclusion of the last assistant message — "
                        "the specific entity, condition, drug, event, or fact it resolved to — "
                        "and use THAT as the referent for any pronouns like 'this', 'it', 'that', 'they'.\n"
                        "2. Do NOT anchor on the document name, document topic, or framing phrases "
                        "like 'the text mentions' or 'according to the document'. "
                        "Anchor on the actual answer: the specific thing the assistant concluded.\n"
                        "3. Preserve the intent of the follow-up exactly — do not add assumptions.\n"
                        "4. Output only the rewritten query. No explanation, no punctuation prefix, nothing else."
                    ),
                ),
                *[ChatMessage(role=m["role"], content=m["content"]) for m in state.history],
                ChatMessage(role="user", content=f"Follow-up: {state.query}"),
            ]
            rewrite_response = self._client.chat(messages=rewrite_messages)
            query = rewrite_response.content or state.query
            print(f"[retrieve] rewritten query: {query}")
        chunks = self._qdrant.query(query, k=self._top_k)

        system_prompt = (
            "You are a knowledgeable assistant that answers questions using a document knowledge base.\n"
            "Rules:\n"
            "1. ALWAYS use your tools before answering. Never answer from memory alone.\n"
            "2. Start with the provided excerpts. If they are incomplete, call get_surrounding_chunks "
            "or get_section_chunks to get more context before resorting to get_full_document.\n"
            "3. Prefer specific tool calls over broad ones — fetch the section before fetching the document.\n"
            "4. Never reveal internal details to the user: no chunk IDs, doc IDs, tool names, retrieval steps, or document structure. Never include chunk IDs, document IDs, or any internal identifiers in your response.\n"
            "5. Answer in clear, natural prose. Be specific and cite facts from the retrieved content.\n"
            "6. If after all tool calls the answer is still not found, reply exactly: I don't know."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system",    "content": system_prompt},
            *self._build_history_context(state.history),
            {"role": "user",      "content": (
            f"Question: {query}\n\n"
            f"Here are the most relevant excerpts retrieved so far. They may be incomplete or lack "
            f"surrounding context — use your tools to fetch more context if needed:\n\n"
            + "\n\n".join(
                f"[chunk_id={c.chunk_id} doc_id={c.doc_id} section_id={c.section_id}]\n{c.content}"
                for c in chunks
                )
            )
            },
        ]

        return {
            "retrieved_chunks": chunks,
            "messages": messages,
            "effective_query": query,
        }

    def _node_generate(self, state: RAGState) -> dict[str, Any]:
        response = self._client.chat_raw(state.messages, tools=MONGO_TOOLS)

        # Append assistant response to message history
        updated_messages = list(state.messages)

        if response.tool_calls:
            updated_messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in response.tool_calls
                ],
            })
        else:
            updated_messages.append({
                "role": "assistant",
                "content": response.content or "",
            })

        return {
            "messages": updated_messages,
            "answer": response.content or "",
        }

    def _node_execute_tool(self, state: RAGState) -> dict[str, Any]:
        """Execute all tool calls from the last assistant message."""
        updated_messages = list(state.messages)
        last = updated_messages[-1]

        for tc in last.get("tool_calls", []):
            name = tc["function"]["name"]
            arguments = tc["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            result = _execute_tool(name, arguments, self._mongo)

            updated_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
                "name": name,
            })

        updated_messages = _trim_tool_results(updated_messages)
        return {
            "messages": updated_messages,
            "tool_call_count": state.tool_call_count + 1,
        }

    def _node_grade(self, state: RAGState) -> dict[str, Any]:
        """Grade how well the answer is grounded in retrieved source material."""
        # Tool results take priority; fall back to initial Qdrant chunks if no tools were called
        tool_contents = [
            msg["content"]
            for msg in state.messages
            if msg.get("role") == "tool"
        ]
        if tool_contents:
            source_text = "\n\n".join(tool_contents)
        else:
            source_text = "\n\n".join(c.content for c in state.retrieved_chunks)

        grade_messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a groundedness grader.\n"
                    "Given a question, an answer, and the source material the model had access to, "
                    "rate from 0.0 to 1.0 how well the answer is grounded in the provided source material.\n"
                    "1.0 means every claim in the answer is directly supported by the source material.\n"
                    "0.0 means the answer ignores the sources entirely and draws on outside knowledge.\n"
                    "Reply with only the number. No explanation, no other text."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Question: {state.effective_query or state.query}\n\n"
                    f"Source material:\n{source_text}\n\n"
                    f"Answer: {state.answer}\n\n"
                    f"Groundedness score (0.0 to 1.0):"
                ),
            ),
        ]

        response = self._client.chat(messages=grade_messages)
        raw = (response.content or "").strip()
        try:
            score = float(raw)
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.5

        return {"confidence_score": score}

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def _edge_after_generate(
        self, state: RAGState
    ) -> Literal["execute_tool", "grade"]:
        """Route to tool execution or grading based on last message."""
        last = state.messages[-1] if state.messages else {}

        has_tool_calls = bool(last.get("tool_calls"))
        over_limit     = state.tool_call_count >= MAX_ITERATIONS

        if has_tool_calls and not over_limit:
            return "execute_tool"
        return "grade"

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        builder: StateGraph = StateGraph(RAGState)

        builder.add_node("retrieve",     self._node_retrieve)
        builder.add_node("generate",     self._node_generate)
        builder.add_node("execute_tool", self._node_execute_tool)
        builder.add_node("grade",        self._node_grade)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve",     "generate")
        builder.add_edge("execute_tool", "generate")
        builder.add_edge("grade",        END)

        builder.add_conditional_edges(
            "generate",
            self._edge_after_generate,
            {
                "execute_tool": "execute_tool",
                "grade":        "grade",
            },
        )

        return builder.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """
        Run the RAG pipeline for a query.

        Returns:
            {
                "answer":         str,
                "confidence_score": float,
                "tool_calls_made": int,
                "sources": [{"chunk_id", "doc_id", "score"}, ...]
            }
        """
        initial_state = RAGState(query=query, history=history or [])
        final_state   = self._graph.invoke(initial_state)

        return {
            "answer":          final_state["answer"],
            "confidence_score": final_state["confidence_score"],
            "tool_calls_made": final_state["tool_call_count"],
            "sources": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id":   c.doc_id,
                    "filename": c.filename,
                    "score":    c.score,
                }
                for c in final_state["retrieved_chunks"]
            ],
        }


# %% Heartbeat

if __name__ == "__main__":
    from config import GroqConfig, HuggingFaceConfig, QdrantConfig, MongoConfig
    from inference_client import InferenceClient
    from IPython.display import Image, display

    client  = InferenceClient.from_config(GroqConfig(), HuggingFaceConfig())
    qdrant  = QdrantStore.from_config(client, QdrantConfig())
    mongo   = MongoStore.from_config(MongoConfig())

    pipeline = RAGPipeline(client=client, qdrant=qdrant, mongo=mongo)

    display(Image(pipeline._graph.get_graph().draw_mermaid_png()))

    result = pipeline.run("What is the product return policy of the company?")

    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence score: {result['confidence_score']}")
    print(f"Tool calls made: {result['tool_calls_made']}")
    print(f"Sources: {len(result['sources'])} chunks retrieved")