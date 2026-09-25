# swgoh-agent

A chat agent for **Star Wars: Galaxy of Heroes** that answers questions about
character and ship kits, team building, counters, Zetas/Omicrons and game
mechanics. Its answers are grounded in game data from [swgoh.gg](https://swgoh.gg).
It works with any **OpenAI-compatible** chat endpoint and includes a terminal
chat and a browser UI.

## How it works

```
swgoh.gg JSON API ──► data/units.json ──► Catalog ─────────┐
 (+ unit pages for       411 units,        names, acronyms,│
  optional base stats)   ~1.9k abilities   filters, effect │
                                           glossary        ├──► tools ──► LangGraph agent ──► CLI / web UI
                              └──► vector index (Chroma) ──┘               (ChatOpenAI)
                                   one doc per ability + per unit,
                                   local fastembed embeddings
```

The agent runs a LangGraph tool-calling loop. The system prompt includes a core
mechanics primer (`src/swgoh_agent/knowledge/mechanics.md`), and the agent has
these tools:

| Tool | Purpose |
| --- | --- |
| `get_unit` | Full kit for a unit: exact ability text, Zeta/Omicron markers (with game mode), ship/crew, base stats, mod data link. Accepts acronyms like `SLKR`, `JMK`, `GAS`. |
| `search_units` | Semantic search over every ability ("who can dispel Taunt?"), with an optional exact `keyword`. |
| `list_units` / `list_categories` | Faction rosters and filters for team building. |
| `lookup_effect` | Unit-specific effects such as Shatterpoint or Payout, as defined in kits. |

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env         # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
uv run swgoh-agent index build   # one-off, ~4 min on CPU; later runs only embed changes
```

`data/units.json` is committed, so you don't need to scrape before a first run.

## Usage

```bash
uv run swgoh-agent serve                 # browser UI at http://127.0.0.1:8000
uv run swgoh-agent chat                  # terminal chat
uv run swgoh-agent chat "What does JMK's ultimate do?"
uv run swgoh-agent chat --log-file agent_reasoning.log   # keep a reasoning trace
uv run swgoh-agent index search "revive Jedi allies"     # test retrieval without the LLM
```

Refresh the game data after a balance update or new release:

```bash
uv run swgoh-agent data refresh               # API only, a few seconds
uv run swgoh-agent data refresh --with-stats  # also scrape base stats (~10 min)
uv run swgoh-agent index build
```

## Configuration

Settings come from environment variables or `.env` (see `.env.example`):

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_BASE_URL` | OpenAI | Any OpenAI-compatible endpoint. `OPENAI_*` names also work. |
| `LLM_API_KEY` | — | |
| `LLM_MODEL` | `gpt-4o-mini` | The model must support tool calling. |
| `LLM_TEMPERATURE` | unset | Left unset because some reasoning models reject it. |
| `EMBEDDING_PROVIDER` | `local` | `local` (fastembed, CPU) or `openai` (an `/embeddings` route). |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Changing it creates a separate index collection. |
| `SWGOH_DATA_DIR` | `./data` | |

## Docker

The image pre-builds the vector index, so it starts ready to serve:

```bash
docker build -t swgoh-agent .
docker run --rm -p 8000:8000 --env-file .env swgoh-agent
```

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Layout

```
src/swgoh_agent/
  config.py            settings (pydantic-settings)
  models.py            Unit / Ability models
  catalog.py           name resolution, filters, effect glossary
  sources/swgohgg.py   swgoh.gg API client + unit page stats scraper
  knowledge/           embeddings, vector index, mechanics primer
  agent/               tools, system prompt, LangGraph agent + event stream
  web/                 FastAPI app (SSE) + single-page UI
  cli.py               click CLI
```
