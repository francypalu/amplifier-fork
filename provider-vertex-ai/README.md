# Amplifier Vertex AI Provider Module

Run Amplifier on Google Vertex AI — Claude **and** Gemini models — using
your GCP project and Application Default Credentials. No API keys.

Inspired by how [open-orbis/open-orbis](https://github.com/open-orbis/open-orbis)
calls Vertex AI for production workloads:

- Claude models go through `anthropic[vertex]`'s `AsyncAnthropicVertex` client.
- Gemini models go through `google-genai` with `vertexai=True`.

The provider auto-routes based on the model id (`claude-*` → Anthropic SDK,
`gemini-*` → google-genai SDK).

## Quick Start

### 1. Authenticate against Google Cloud

```bash
# Local dev
gcloud auth application-default login

# Set your project
export GOOGLE_CLOUD_PROJECT="my-gcp-project"
export GOOGLE_CLOUD_LOCATION="europe-west1"   # any Vertex region
```

In Cloud Run / GKE, ADC is provided automatically by the workload-identity
service account — no env var needed beyond the project id.

Make sure the Vertex AI models you intend to use are enabled in the
[Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden).

### 2. Add the module

```bash
amplifier module add provider-vertex-ai \
  --source git+https://github.com/<your-fork>/amplifier@providers/vertex_ai#subdirectory=provider-vertex-ai \
  --global
```

### 3. Use Vertex AI as your provider

```bash
amplifier provider use vertex-ai --model claude-sonnet-4-5@20250929
# or
amplifier provider use vertex-ai --model gemini-2.5-pro
```

### 4. Or via a bundle

```yaml
---
bundle:
  name: vertex-dev
  version: 0.1.0

includes:
  - bundle: foundation

providers:
  - module: provider-vertex-ai
    source: git+https://github.com/<your-fork>/amplifier@providers/vertex_ai#subdirectory=provider-vertex-ai
    config:
      project_id: my-gcp-project
      location: europe-west1
      default_model: claude-sonnet-4-5@20250929
      max_tokens: 16000
      temperature: 0.7
      priority: 50
---
```

## Supported models

| Model id | Family | Notes |
|---|---|---|
| `claude-opus-4-1@20250805` | Claude Opus 4.1 | Best reasoning |
| `claude-sonnet-4-5@20250929` | Claude Sonnet 4.5 | Default |
| `claude-haiku-4-5@20251001` | Claude Haiku 4.5 | Fast |
| `gemini-2.5-pro` | Gemini 2.5 Pro | 2M context |
| `gemini-2.5-flash` | Gemini 2.5 Flash | 1M context, fast |

Add or override the list at runtime by extending `VertexAIProvider.list_models`
or by passing any model id Vertex accepts via `--model`.

## Configuration

| Field | Env var | Default | Notes |
|---|---|---|---|
| `project_id` | `GOOGLE_CLOUD_PROJECT` | — | Required |
| `location` | `GOOGLE_CLOUD_LOCATION` | `us-central1` | Any Vertex AI region |
| `default_model` | — | `claude-sonnet-4-5@20250929` | Used when no `--model` flag |
| `max_tokens` | — | `8192` | |
| `temperature` | — | `0.7` | |
| `timeout` | — | `600.0` (s) | Per-request timeout |
| `priority` | — | `100` | Lower = preferred when multiple providers are mounted |

## Why ADC instead of API keys?

Vertex AI doesn't issue API keys — it expects bearer tokens minted from a
Google identity. Using ADC means:

- The same code runs locally (`gcloud auth application-default login`) and
  in Cloud Run / GKE (workload identity) without changes.
- Billing, quotas, and audit logs land on your GCP project.
- Secrets never need to be stored in `.env` or shared between developers.

## Layout

```
provider-vertex-ai/
├── pyproject.toml
├── README.md
└── amplifier_module_provider_vertex_ai/
    └── __init__.py        # mount() + VertexAIProvider
```

The package follows the same `mount(coordinator, config)` contract as the
other Amplifier providers (`provider-anthropic`, `provider-gemini`,
`provider-bedrock`), so it slots into the standard module-loading
machinery without changes to amplifier-core or amplifier-foundation.
