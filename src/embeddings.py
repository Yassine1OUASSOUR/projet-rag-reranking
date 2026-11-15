"""Embeddings E5 : les préfixes 'query:' / 'passage:' sont obligatoires pour de bons résultats."""
from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from .config import SETTINGS, get_device


class E5Embeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


def get_embeddings() -> E5Embeddings:
    return E5Embeddings(
        model_name=SETTINGS.embedding_model,
        model_kwargs={"device": get_device()},
        encode_kwargs={"normalize_embeddings": True},
    )