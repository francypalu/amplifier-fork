"""Google Vertex AI provider module for Amplifier.

Routes requests to Claude or Gemini models hosted on Google Vertex AI.

- Claude models (model id starts with ``claude-``) go through the
  ``anthropic[vertex]`` SDK using ``AsyncAnthropicVertex``.
- Gemini models (model id starts with ``gemini-``) go through the
  ``google-genai`` SDK with ``vertexai=True``.

Authentication uses Application Default Credentials (ADC). On Cloud Run /
GKE this is provided automatically by the workload identity service
account; locally, run ``gcloud auth application-default login`` once.

Inspired by:
- microsoft/amplifier-module-provider-anthropic   (provider contract)
- microsoft/amplifier-module-provider-gemini      (Gemini call shape)
- brycecutt-msft/amplifier-module-provider-bedrock (cloud-hosted Claude)
- open-orbis/open-orbis                            (Vertex AI integration)
"""

from __future__ import annotations

__all__ = ["mount", "VertexAIProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import logging
import os
import time
from typing import Any

from amplifier_core import (
    ConfigField,
    ModelInfo,
    ModuleCoordinator,
    ProviderInfo,
)
from amplifier_core.content_models import (
    TextContent,
    ThinkingContent,
    ToolCallContent,
)
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    Message,
    TextBlock,
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------


async def mount(
    coordinator: ModuleCoordinator, config: dict[str, Any] | None = None
):
    """Mount the Vertex AI provider.

    Returns ``None`` (graceful degradation) when no GCP project is
    configured — the provider can't function without one.
    """
    config = config or {}

    provider = VertexAIProvider(
        api_key=None, config=config, coordinator=coordinator
    )

    if not provider.project_id:
        logger.warning(
            "Vertex AI provider not mounted: no GCP project configured. "
            "Set GOOGLE_CLOUD_PROJECT (or pass project_id in config), then "
            "run `gcloud auth application-default login` for local dev."
        )
        return None

    provider_name = config.get("name", "vertex-ai")
    await coordinator.mount("providers", provider, name=provider_name)
    logger.info(
        "Mounted VertexAIProvider as '%s' (project=%s, location=%s)",
        provider_name,
        provider.project_id,
        provider.location,
    )

    async def cleanup():
        await provider.close()

    return cleanup


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class VertexAIProvider:
    """Google Vertex AI integration for Claude and Gemini models."""

    name = "vertex-ai"
    api_label = "Vertex AI"

    def __init__(
        self,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ):
        # Vertex AI uses Application Default Credentials, not API keys.
        # We accept ``api_key`` only to match the signature amplifier's
        # provider auto-instantiation tries (same as Anthropic / Gemini).
        del api_key
        self.config = config or {}
        self.coordinator = coordinator
        self.project_id = (
            self.config.get("project_id")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT_ID")
            or ""
        )
        self.location = (
            self.config.get("location")
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or os.environ.get("VERTEX_REGION")
            or "us-central1"
        )

        self.default_model = self.config.get(
            "default_model", "claude-sonnet-4-5@20250929"
        )
        self.max_tokens = int(self.config.get("max_tokens", 8192))
        self.temperature = float(self.config.get("temperature", 0.7))
        self.timeout = float(self.config.get("timeout", 600.0))
        self.priority = int(self.config.get("priority", 100))

        # Lazy-initialised SDK clients — built on first use so get_info()
        # can run before credentials are resolved.
        self._anthropic_client = None
        self._genai_client = None

    # ---- backend dispatch -------------------------------------------------

    @staticmethod
    def _backend_for(model: str) -> str:
        """Return ``"claude"`` or ``"gemini"`` based on the model id."""
        m = model.lower()
        if m.startswith("claude") or "claude" in m:
            return "claude"
        if m.startswith("gemini") or "gemini" in m:
            return "gemini"
        raise ValueError(
            f"Unknown Vertex AI model family for '{model}'. "
            "Expected an id starting with 'claude-' or 'gemini-'."
        )

    @property
    def anthropic_client(self):
        """Lazily build the AsyncAnthropicVertex client."""
        if self._anthropic_client is None:
            from anthropic import AsyncAnthropicVertex

            self._anthropic_client = AsyncAnthropicVertex(
                project_id=self.project_id,
                region=self.location,
            )
        return self._anthropic_client

    @property
    def genai_client(self):
        """Lazily build the google-genai Vertex client."""
        if self._genai_client is None:
            from google import genai

            self._genai_client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
        return self._genai_client

    # ---- module contract --------------------------------------------------

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            id="vertex-ai",
            display_name="Google Vertex AI",
            credential_env_vars=[
                "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_APPLICATION_CREDENTIALS",
            ],
            capabilities=["streaming", "tools", "thinking", "vision"],
            defaults={
                "model": self.default_model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "timeout": self.timeout,
            },
            config_fields=[
                ConfigField(
                    id="project_id",
                    display_name="GCP Project ID",
                    field_type="text",
                    prompt="Google Cloud project ID",
                    env_var="GOOGLE_CLOUD_PROJECT",
                    required=True,
                ),
                ConfigField(
                    id="location",
                    display_name="Vertex AI Region",
                    field_type="text",
                    prompt="Vertex AI region (e.g. us-central1, europe-west1)",
                    env_var="GOOGLE_CLOUD_LOCATION",
                    default="us-central1",
                    required=False,
                ),
                ConfigField(
                    id="default_model",
                    display_name="Default Model",
                    field_type="text",
                    prompt="Default model id (claude-* or gemini-*)",
                    default="claude-sonnet-4-5@20250929",
                    required=False,
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return the curated list of Vertex AI Claude + Gemini models.

        Vertex's Claude catalog is announced via Google docs rather than a
        live discovery endpoint, so we hard-code the mainstream ids here
        (mirrors how the Bedrock provider does it). Gemini ids are listed
        alongside; the live ``genai_client.models.list`` call returns the
        full catalog if you'd rather discover them at runtime.
        """
        return [
            # ---- Claude on Vertex AI ----
            ModelInfo(
                id="claude-opus-4-1@20250805",
                display_name="Claude Opus 4.1 (Vertex)",
                context_window=200000,
                max_output_tokens=32000,
                capabilities=["tools", "thinking", "vision", "streaming"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="claude-sonnet-4-5@20250929",
                display_name="Claude Sonnet 4.5 (Vertex)",
                context_window=200000,
                max_output_tokens=16000,
                capabilities=["tools", "thinking", "vision", "streaming"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="claude-haiku-4-5@20251001",
                display_name="Claude Haiku 4.5 (Vertex)",
                context_window=200000,
                max_output_tokens=8192,
                capabilities=["tools", "vision", "streaming", "fast"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            # ---- Gemini on Vertex AI ----
            ModelInfo(
                id="gemini-2.5-pro",
                display_name="Gemini 2.5 Pro (Vertex)",
                context_window=2000000,
                max_output_tokens=65536,
                capabilities=["tools", "thinking", "vision", "streaming"],
                defaults={"temperature": 0.7, "max_tokens": 8192},
            ),
            ModelInfo(
                id="gemini-2.5-flash",
                display_name="Gemini 2.5 Flash (Vertex)",
                context_window=1048576,
                max_output_tokens=65536,
                capabilities=["tools", "thinking", "vision", "streaming", "fast"],
                defaults={"temperature": 0.7, "max_tokens": 8192},
            ),
        ]

    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Dispatch the request to the correct backend based on the model id."""
        model = str(kwargs.get("model", self.default_model))
        backend = self._backend_for(model)

        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": self.name,
                    "backend": backend,
                    "model": model,
                    "message_count": len(request.messages),
                },
            )

        start = time.time()
        if backend == "claude":
            response = await self._complete_claude(request, model, **kwargs)
        else:
            response = await self._complete_gemini(request, model, **kwargs)

        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(
                "llm:response",
                {
                    "provider": self.name,
                    "backend": backend,
                    "model": model,
                    "elapsed_ms": int((time.time() - start) * 1000),
                },
            )
        return response

    async def close(self) -> None:
        """Close any open SDK clients."""
        if self._anthropic_client is not None and hasattr(
            self._anthropic_client, "close"
        ):
            try:
                await self._anthropic_client.close()
            except Exception:  # SDK close is best-effort
                logger.debug("AsyncAnthropicVertex.close() raised", exc_info=True)
        # google-genai's client doesn't expose an async close; nothing to do.

    # ---- Claude backend ---------------------------------------------------

    async def _complete_claude(
        self, request: ChatRequest, model: str, **kwargs
    ) -> ChatResponse:
        """Call Claude on Vertex via the Anthropic SDK."""
        system_msgs = [m for m in request.messages if m.role == "system"]
        conversation = [
            m for m in request.messages if m.role in ("user", "assistant", "tool")
        ]

        system = (
            "\n\n".join(
                m.content if isinstance(m.content, str) else "" for m in system_msgs
            )
            if system_msgs
            else None
        )

        params: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in conversation],
            "max_tokens": request.max_output_tokens
            or kwargs.get("max_tokens", self.max_tokens),
            "temperature": request.temperature
            or kwargs.get("temperature", self.temperature),
        }
        if system:
            params["system"] = system
        if request.tools:
            params["tools"] = self._convert_tools(request.tools)

        response = await asyncio.wait_for(
            self.anthropic_client.messages.create(**params),
            timeout=self.timeout,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
        )

        return ChatResponse(
            message=Message(
                role="assistant",
                content=[TextBlock(text="".join(text_parts))]
                if text_parts
                else "",
            ),
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            stop_reason=getattr(response, "stop_reason", None),
        )

    # ---- Gemini backend ---------------------------------------------------

    async def _complete_gemini(
        self, request: ChatRequest, model: str, **kwargs
    ) -> ChatResponse:
        """Call Gemini on Vertex via the google-genai SDK."""
        from google import genai

        system_msgs = [m for m in request.messages if m.role == "system"]
        conversation = [
            m for m in request.messages if m.role in ("user", "assistant", "tool")
        ]

        system_instruction = (
            "\n\n".join(
                m.content if isinstance(m.content, str) else "" for m in system_msgs
            )
            if system_msgs
            else None
        )

        # Vertex Gemini wants {role, parts:[{text}]} — ``user`` for user/tool,
        # ``model`` for assistant. This is the same convention the Microsoft
        # Gemini provider uses.
        contents = []
        for m in conversation:
            text = m.content if isinstance(m.content, str) else ""
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": text}]})

        config = genai.types.GenerateContentConfig(
            temperature=request.temperature
            or kwargs.get("temperature", self.temperature),
            max_output_tokens=request.max_output_tokens
            or kwargs.get("max_tokens", self.max_tokens),
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = await asyncio.wait_for(
            self.genai_client.aio.models.generate_content(
                model=model, contents=contents, config=config
            ),
            timeout=self.timeout,
        )

        text = response.text or ""
        usage = Usage(
            input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0)
            if response.usage_metadata
            else 0,
            output_tokens=getattr(
                response.usage_metadata, "candidates_token_count", 0
            )
            if response.usage_metadata
            else 0,
        )
        return ChatResponse(
            message=Message(
                role="assistant",
                content=[TextBlock(text=text)] if text else "",
            ),
            tool_calls=[],
            usage=usage,
            model=model,
            stop_reason=None,
        )

    # ---- shared helpers ---------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
        """Translate Amplifier tool descriptors to the Anthropic tool schema."""
        out: list[dict[str, Any]] = []
        for t in tools:
            spec: dict[str, Any] = {
                "name": getattr(t, "name", None) or t["name"],
                "description": getattr(t, "description", None)
                or t.get("description", ""),
                "input_schema": getattr(t, "parameters", None)
                or t.get("parameters", {"type": "object", "properties": {}}),
            }
            out.append(spec)
        return out
