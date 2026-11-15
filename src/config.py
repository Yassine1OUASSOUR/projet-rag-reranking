"""Configuration centralisée (surchargeable par variables d'environnement)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "data"))
    chroma_dir: Path = Path(os.getenv("RAG_CHROMA_DIR", "chroma_db"))
    collection: str = "rag_corpus"

    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_chars: int = 80

    rrf_k: int = 60            # constante de la Reciprocal Rank Fusion
    candidate_pool: int = 30   # candidats envoyés au cross-encoder

    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")


SETTINGS = Settings()