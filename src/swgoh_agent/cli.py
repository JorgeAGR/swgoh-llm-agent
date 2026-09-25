"""Command line interface: data refresh, index build, terminal chat and web server."""

import asyncio
import logging
import uuid
from pathlib import Path

import click
from tqdm import tqdm

from swgoh_agent.config import get_settings


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """A SWGOH chat agent grounded in swgoh.gg game data."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


# --- data ------------------------------------------------------------------


@cli.group()
def data() -> None:
    """Manage the local unit dataset."""


@data.command("refresh")
@click.option(
    "--with-stats",
    is_flag=True,
    help="Also scrape base stats from every unit page (slow; ~1 request/sec).",
)
@click.option("--delay", default=1.0, show_default=True, help="Seconds between pages.")
def data_refresh(with_stats: bool, delay: float) -> None:
    """Download characters, ships and abilities from swgoh.gg."""
    from swgoh_agent.catalog import Catalog
    from swgoh_agent.sources.swgohgg import (
        SwgohGGClient,
        fetch_base_stats,
        fetch_dataset,
    )

    settings = get_settings()
    client = SwgohGGClient(delay=delay)
    click.echo("Fetching units and abilities from the swgoh.gg API…")
    dataset = fetch_dataset(client)

    if with_stats:
        with tqdm(total=len(dataset.units), desc="Base stats", unit="unit") as bar:
            fetch_base_stats(client, dataset.units, on_progress=lambda _: bar.update())
    elif settings.units_path.exists():
        # Keep previously scraped stats rather than dropping them.
        previous = Catalog.load(settings.units_path)
        for unit in dataset.units:
            if old := previous.get(unit.base_id):
                unit.base_stats = old.base_stats

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.units_path.write_text(dataset.model_dump_json(indent=1))
    n_abilities = sum(len(u.abilities) for u in dataset.units)
    click.echo(
        f"Saved {len(dataset.units)} units ({n_abilities} abilities) "
        f"to {settings.units_path}."
    )
    click.echo("Run `swgoh-agent index build` to update the knowledge index.")


# --- index -----------------------------------------------------------------


@cli.group()
def index() -> None:
    """Manage the vector index used for semantic search."""


@index.command("build")
def index_build() -> None:
    """Embed new or changed abilities and drop stale ones."""
    from swgoh_agent.catalog import Catalog
    from swgoh_agent.knowledge.index import KnowledgeIndex, build_documents

    settings = get_settings()
    catalog = Catalog.load(settings.units_path)
    docs = build_documents(catalog)
    click.echo(f"Indexing {len(docs)} documents with {settings.embedding_model}…")
    knowledge = KnowledgeIndex(settings)
    with tqdm(desc="Embedding", unit="doc") as bar:
        added, removed = knowledge.sync(docs, progress=bar.update)
    click.echo(f"Added {added}, removed {removed}; index holds {knowledge.count()}.")


@index.command("search")
@click.argument("query")
@click.option("-k", "--keyword", help="Require this exact term in the kit text.")
def index_search(query: str, keyword: str | None) -> None:
    """Try a semantic search without involving the LLM."""
    from swgoh_agent.knowledge.index import KnowledgeIndex

    for hit in KnowledgeIndex(get_settings()).search_units(query, keyword=keyword):
        click.echo(f"{hit.score:.2f}  {hit.unit}  [{', '.join(hit.abilities[:3])}]")


# --- chat ------------------------------------------------------------------


def _attach_trace_log(path: Path) -> None:
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    trace = logging.getLogger("swgoh_agent.trace")
    trace.setLevel(logging.INFO)
    trace.addHandler(handler)


@cli.command()
@click.argument("question", required=False)
@click.option("--show-tools/--hide-tools", default=True, help="Print tool calls.")
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Append the agent's reasoning trace to this file.",
)
def chat(question: str | None, show_tools: bool, log_file: Path | None) -> None:
    """Chat in the terminal. Pass QUESTION for a one-shot answer."""
    from swgoh_agent.agent.graph import create_agent

    if log_file:
        _attach_trace_log(log_file)
    agent = create_agent(get_settings())
    thread_id = uuid.uuid4().hex

    async def turn(message: str) -> None:
        async for event in agent.astream(message, thread_id):
            if event["type"] == "token":
                click.echo(event["text"], nl=False)
            elif event["type"] == "tool_call" and show_tools:
                args = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
                click.secho(f"  ⚙ {event['name']}({args})", fg="bright_black")
        click.echo()

    async def repl() -> None:
        click.echo("SWGOH tactical droid online. Ctrl-D or 'exit' to quit.\n")
        prompt = click.style("you", fg="cyan")
        while True:
            try:
                message = await asyncio.to_thread(
                    click.prompt, prompt, prompt_suffix="> "
                )
            except (EOFError, click.Abort):
                break
            if message.strip().lower() in {"exit", "quit"}:
                break
            await turn(message)

    # One event loop for the whole session: the HTTP client is bound to it.
    asyncio.run(turn(question) if question else repl())


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Append the agent's reasoning trace to this file.",
)
def serve(host: str, port: int, log_file: Path | None) -> None:
    """Serve the browser chat interface."""
    import uvicorn

    from swgoh_agent.web.app import create_app

    if log_file:
        _attach_trace_log(log_file)
    click.echo(f"Serving on http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)
