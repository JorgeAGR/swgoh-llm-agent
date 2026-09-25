"""Vector index over unit profiles and individual abilities."""

import hashlib
import logging
import re
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document

from swgoh_agent.catalog import Catalog
from swgoh_agent.config import Settings
from swgoh_agent.knowledge.embeddings import get_embeddings
from swgoh_agent.models import Unit

log = logging.getLogger(__name__)

_BATCH_SIZE = 256


def _header(unit: Unit) -> str:
    return (
        f"Unit: {unit.name} ({unit.combat_type}; {unit.alignment} {unit.role})\n"
        f"Categories: {', '.join(unit.categories)}"
    )


def _metadata(unit: Unit, kind: str, ability: str = "") -> dict:
    return {
        "base_id": unit.base_id,
        "unit": unit.name,
        "kind": kind,
        "ability": ability,
        "alignment": unit.alignment,
        "combat_type": unit.combat_type,
    }


def unit_documents(unit: Unit) -> list[Document]:
    profile = (
        f"{_header(unit)}\n{unit.description}\n"
        f"Ability classes: {', '.join(unit.ability_classes)}"
    )
    docs = [Document(page_content=profile, metadata=_metadata(unit, "profile"))]
    for ability in unit.abilities:
        docs.append(
            Document(
                page_content=f"{_header(unit)}\n{ability.render()}",
                metadata=_metadata(unit, "ability", ability.name),
            )
        )
    return docs


def build_documents(catalog: Catalog) -> list[Document]:
    return [doc for unit in catalog.units for doc in unit_documents(unit)]


def _doc_id(doc: Document) -> str:
    return hashlib.sha256(doc.page_content.encode()).hexdigest()[:32]


@dataclass
class UnitHit:
    unit: str
    base_id: str
    score: float
    abilities: list[str] = field(default_factory=list)


class KnowledgeIndex:
    def __init__(self, settings: Settings):
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", settings.embedding_model).strip("-")
        self.store = Chroma(
            collection_name=f"units-{slug}".lower()[:63],
            embedding_function=get_embeddings(settings),
            persist_directory=str(settings.index_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.store._collection.count()

    def sync(self, docs: list[Document], progress=None) -> tuple[int, int]:
        """Embed new/changed documents, drop stale ones; return (added, removed)."""
        wanted = {_doc_id(d): d for d in docs}
        existing = set(self.store.get(include=[])["ids"])
        stale = list(existing - wanted.keys())
        new_ids = [i for i in wanted if i not in existing]
        if stale:
            self.store.delete(ids=stale)
        for start in range(0, len(new_ids), _BATCH_SIZE):
            batch = new_ids[start : start + _BATCH_SIZE]
            self.store.add_documents([wanted[i] for i in batch], ids=batch)
            if progress:
                progress(len(batch))
        return len(new_ids), len(stale)

    def search_units(
        self,
        query: str,
        limit: int = 8,
        alignment: str | None = None,
        combat_type: str | None = None,
        keyword: str | None = None,
    ) -> list[UnitHit]:
        """Semantic search over abilities and profiles, aggregated per unit.

        `keyword` restricts results to documents containing that literal term,
        which keeps precise game vocabulary (e.g. "Shatterpoint") from drifting.
        """
        filters = [
            {key: value}
            for key, value in (("alignment", alignment), ("combat_type", combat_type))
            if value
        ]
        where = (
            {"$and": filters} if len(filters) > 1 else (filters[0] if filters else None)
        )
        where_document = None
        if keyword:
            # Chroma's $contains is case-sensitive; cover the common casings.
            variants = list(dict.fromkeys([keyword, keyword.lower(), keyword.title()]))
            where_document = (
                {"$or": [{"$contains": v} for v in variants]}
                if len(variants) > 1
                else {"$contains": keyword}
            )
        results = self.store.similarity_search_with_score(
            query, k=limit * 5, filter=where, where_document=where_document
        )
        hits: dict[str, UnitHit] = {}
        for doc, distance in results:
            meta = doc.metadata
            hit = hits.setdefault(
                meta["base_id"],
                UnitHit(meta["unit"], meta["base_id"], score=1 - distance),
            )
            if meta["ability"] and meta["ability"] not in hit.abilities:
                hit.abilities.append(meta["ability"])
        return list(hits.values())[:limit]
