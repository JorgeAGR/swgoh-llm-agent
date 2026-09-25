"""Embedding backends: a local ONNX model or an OpenAI-compatible endpoint."""

from langchain_core.embeddings import Embeddings

from swgoh_agent.config import Settings


class FastEmbedEmbeddings(Embeddings):
    """Runs a small embedding model locally, so no embeddings API is required."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "local":
        return FastEmbedEmbeddings(
            settings.embedding_model, cache_dir=str(settings.model_cache_dir)
        )
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            base_url=settings.embedding_base_url or settings.llm_base_url,
            api_key=settings.embedding_api_key or settings.llm_api_key,
            # Non-OpenAI servers expect raw strings, not tiktoken token ids.
            check_embedding_ctx_length=False,
        )
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER {settings.embedding_provider!r}; "
        "expected 'local' or 'openai'."
    )
