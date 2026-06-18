# %% Imports
from __future__ import annotations

from typing import Any, Generator, cast

from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import GroqConfig, HuggingFaceConfig

# ---------------------------------------------------------------------------
# Pydantic models  (same public shape as the old ollama_client.py)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class AssistantMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @classmethod
    def from_groq(cls, raw_message: Any) -> "AssistantMessage":
        """
        Parse a Groq ChatCompletionMessage into our internal model.
        Groq tool_calls look like:
          [ChoiceDeltaToolCall(id=..., function=Function(name=..., arguments='{"k": v}'))]
        """
        import json

        tool_calls: list[ToolCall] | None = None

        if raw_message.tool_calls:
            tool_calls = []
            for tc in raw_message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return cls(
            role=raw_message.role,
            content=raw_message.content or None,
            tool_calls=tool_calls,
        )


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "function": self.function.to_dict()}


# ---------------------------------------------------------------------------
# InferenceClient  (drop-in replacement for OllamaClient)
# ---------------------------------------------------------------------------


class InferenceClient:
    """
    Unified client for:
      - Chat / generation  →  Groq API  (fast cloud LLM inference)
      - Embeddings         →  HuggingFace sentence-transformers (local)

    Drop-in replacement for OllamaClient; exposes the same public methods:
        embed(), embed_batch(), generate(), generate_stream(), chat(), chat_stream(), is_available()
    """

    def __init__(
        self,
        groq_api_key: str,
        groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.llm_model = groq_model
        self.embedding_model = embedding_model_name

        # Groq client (chat + generation)
        self._groq = Groq(api_key=groq_api_key)

        # HuggingFace sentence-transformers (embeddings, loaded once)
        print(f"[inference] Loading embedding model: {embedding_model_name} ...")
        self._st_model = SentenceTransformer(embedding_model_name, device=device)
        print(f"[inference] Embedding model ready.")

    @classmethod
    def from_config(
        cls,
        groq_cfg: GroqConfig,
        hf_cfg: HuggingFaceConfig,
    ) -> "InferenceClient":
        return cls(
            groq_api_key=groq_cfg.api_key,
            groq_model=groq_cfg.model,
            embedding_model_name=hf_cfg.model_name,
            device=hf_cfg.device,
        )

    # ------------------------------------------------------------------
    # Embeddings  (HuggingFace sentence-transformers)
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        vector = self._st_model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings (batched for efficiency)."""
        vectors = self._st_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    # ------------------------------------------------------------------
    # Generation  (Groq)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Single-turn generation. Returns the full response string."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._groq.chat.completions.create(
            model=self.llm_model,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""

    def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
    ) -> Generator[str, None, None]:
        """Streaming single-turn generation. Yields text chunks as they arrive."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = self._groq.chat.completions.create(
            model=self.llm_model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                yield text

    # ------------------------------------------------------------------
    # Chat  (multi-turn, with optional tool calls)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] | None = None,
    ) -> AssistantMessage:
        """
        Multi-turn chat with optional tool call support.

        Groq supports OpenAI-compatible tool_choice; we pass tools in the
        same dict format and parse the response back into AssistantMessage.
        """
        serialised: list[dict[str, Any]] = [m.to_dict() for m in messages]

        kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "messages": serialised,
        }
        if tools:
            kwargs["tools"] = [t.to_dict() for t in tools]
            kwargs["tool_choice"] = "auto"

        response = self._groq.chat.completions.create(**kwargs)
        return AssistantMessage.from_groq(response.choices[0].message)

    def chat_raw(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
    ) -> AssistantMessage:
        """Like chat() but accepts pre-serialised dicts directly."""
        import groq as groq_module

        kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = [t.to_dict() for t in tools]
            kwargs["tool_choice"] = "auto"

        try:
            response = self._groq.chat.completions.create(**kwargs)
        except groq_module.BadRequestError as e:
            if "tool_use_failed" in str(e):
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                response = self._groq.chat.completions.create(**kwargs)
            else:
                raise

        return AssistantMessage.from_groq(response.choices[0].message)

    def chat_stream(
        self,
        messages: list[ChatMessage],
    ) -> Generator[str, None, None]:
        """Streaming multi-turn chat (no tool calls). Yields text chunks."""
        serialised: list[dict[str, Any]] = [m.to_dict() for m in messages]

        stream = self._groq.chat.completions.create(
            model=self.llm_model,
            messages=serialised,  # type: ignore[arg-type]
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                yield text

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Returns True if Groq is reachable and the embedding model is loaded."""
        try:
            # Quick Groq ping via a minimal completion
            self._groq.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            # Embedding model is local — if __init__ succeeded it's loaded
            return True
        except Exception:
            return False

