"""Smoke tests for the Vertex AI provider module.

These tests do not hit Vertex AI; they validate the local contract:
- ``mount()`` declines gracefully when no project is configured.
- ``mount()`` registers the provider when a project is configured.
- Backend dispatch picks Claude vs Gemini from the model id.
- ``get_info()`` and ``list_models()`` return well-formed metadata.
"""

from __future__ import annotations

import pytest

from amplifier_module_provider_vertex_ai import VertexAIProvider, mount


class _FakeCoordinator:
    def __init__(self):
        self.mounted: list[tuple[str, object, str]] = []

    async def mount(self, kind, instance, name):
        self.mounted.append((kind, instance, name))


@pytest.mark.asyncio
async def test_mount_skips_without_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    coord = _FakeCoordinator()

    cleanup = await mount(coord, {})

    assert cleanup is None
    assert coord.mounted == []


@pytest.mark.asyncio
async def test_mount_registers_provider(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-proj")
    coord = _FakeCoordinator()

    cleanup = await mount(coord, {"location": "europe-west1"})

    assert cleanup is not None
    assert len(coord.mounted) == 1
    kind, provider, name = coord.mounted[0]
    assert kind == "providers"
    assert name == "vertex-ai"
    assert isinstance(provider, VertexAIProvider)
    assert provider.project_id == "my-proj"
    assert provider.location == "europe-west1"


def test_backend_routing():
    assert VertexAIProvider._backend_for("claude-sonnet-4-5@20250929") == "claude"
    assert VertexAIProvider._backend_for("claude-opus-4-1@20250805") == "claude"
    assert VertexAIProvider._backend_for("gemini-2.5-pro") == "gemini"
    assert VertexAIProvider._backend_for("gemini-2.5-flash") == "gemini"

    with pytest.raises(ValueError):
        VertexAIProvider._backend_for("gpt-5.5")


def test_get_info_shape():
    p = VertexAIProvider(project_id="x", location="us-central1")
    info = p.get_info()
    assert info.id == "vertex-ai"
    assert "GOOGLE_CLOUD_PROJECT" in info.credential_env_vars
    field_ids = {f.id for f in info.config_fields}
    assert {"project_id", "location", "default_model"} <= field_ids


@pytest.mark.asyncio
async def test_list_models_returns_both_families():
    p = VertexAIProvider(project_id="x", location="us-central1")
    models = await p.list_models()
    ids = [m.id for m in models]
    assert any(i.startswith("claude-") for i in ids)
    assert any(i.startswith("gemini-") for i in ids)
