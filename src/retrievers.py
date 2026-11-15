"""Moteur de recherche comparatif : Dense, Hybride (BM25 + Dense via RRF) et Reranking cross-encoder."""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import torch
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import SETTINGS, get_device
from .embeddings import get_embeddings

_STOPWORDS = set(
    "le la les un une des de du et en au aux ce ces que qui quoi dans par pour sur est sont "
    "a an the of and or in on to is are be as at by for from with that this it its what which how".split()
)


def tokenize(text: str) -> list[str]:
    """Minuscules, sans accents, sans mots vides : 'RGPD' == 'rgpd', 'définition' == 'definition'."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"\w+", text) if len(t) > 1 and t not in _STOPWORDS]


@dataclass
class Hit:
    text: str
    metadata: dict
    score: float
    rank: int = 0
    origin: dict = field(default_factory=dict)  # rang d'origine (dense / bm25) pour l'hybride

    @property
    def key(self) -> str:
        return hashlib.md5(self.text.encode()).hexdigest()

    @property
    def source(self) -> str:
        return self.metadata.get("source", "?")

    @property
    def page(self):
        return self.metadata.get("page", "?")


@dataclass
class SearchResult:
    strategy: str
    hits: list[Hit]
    latency_ms: float


class RAGComparator:
    def __init__(self) -> None:
        self.store = Chroma(
            collection_name=SETTINGS.collection,
            embedding_function=get_embeddings(),
            persist_directory=str(SETTINGS.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        data = self.store.get(include=["documents", "metadatas"])
        self.texts: list[str] = data["documents"]
        self.metadatas: list[dict] = data["metadatas"]
        if not self.texts:
            raise RuntimeError("Base vide : lancez d'abord `python -m src.ingestion`.")
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.device = get_device()

    # -- ressources lourdes chargées à la demande ---------------------------------------------
    @cached_property
    def _tokenizer(self):
        return AutoTokenizer.from_pretrained(SETTINGS.reranker_model)

    @cached_property
    def _reranker(self):
        model = AutoModelForSequenceClassification.from_pretrained(SETTINGS.reranker_model)
        return model.to(self.device).eval()

    # -- informations sur le corpus -------------------------------------------------------------
    @property
    def sources(self) -> list[str]:
        return sorted({m.get("source", "?") for m in self.metadatas})

    def corpus_summary(self) -> list[dict]:
        rows = []
        for src in self.sources:
            idx = [i for i, m in enumerate(self.metadatas) if m.get("source") == src]
            rows.append(
                {
                    "Document": src,
                    "Pages": len({self.metadatas[i].get("page") for i in idx}),
                    "Chunks": len(idx),
                    "Taille moyenne (car.)": int(np.mean([len(self.texts[i]) for i in idx])),
                }
            )
        return rows

    # -- stratégies -----------------------------------------------------------------------------
    def dense(self, query: str, k: int = 5, sources: list[str] | None = None) -> list[Hit]:
        flt = {"source": {"$in": sources}} if sources else None
        results = self.store.similarity_search_with_score(query, k=k, filter=flt)
        # distance cosinus -> similarité
        return [Hit(d.page_content, d.metadata, 1.0 - dist, rank=i) for i, (d, dist) in enumerate(results, 1)]

    def bm25_search(self, query: str, k: int = 5, sources: list[str] | None = None) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        if sources:
            mask = np.array([m.get("source") in sources for m in self.metadatas])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [
            Hit(self.texts[i], self.metadatas[i], float(scores[i]), rank=r)
            for r, i in enumerate(order, 1)
            if scores[i] > 0
        ]

    def hybrid(
        self,
        query: str,
        k: int = 5,
        pool: int = 30,
        bm25_query: str | None = None,
        sources: list[str] | None = None,
    ) -> list[Hit]:
        """Fusion RRF : score(d) = Σ 1 / (k_rrf + rang(d)). `bm25_query` permet une requête traduite."""
        ranked = {
            "dense": self.dense(query, pool, sources),
            "bm25": self.bm25_search(bm25_query or query, pool, sources),
        }
        fused: dict[str, Hit] = {}
        for name, hits in ranked.items():
            for hit in hits:
                entry = fused.setdefault(hit.key, Hit(hit.text, hit.metadata, 0.0))
                entry.score += 1.0 / (SETTINGS.rrf_k + hit.rank)
                entry.origin[name] = hit.rank
        out = sorted(fused.values(), key=lambda h: h.score, reverse=True)[:k]
        for i, hit in enumerate(out, 1):
            hit.rank = i
        return out

    @torch.no_grad()
    def rerank(self, query: str, candidates: list[Hit], top_k: int = 5, batch_size: int = 8) -> list[Hit]:
        if not candidates:
            return []
        scores: list[float] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            inputs = self._tokenizer(
                [[query, c.text] for c in batch],
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            ).to(self.device)
            logits = self._reranker(**inputs).logits.view(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:top_k]
        out = []
        for rank, i in enumerate(order, 1):
            c = candidates[i]
            out.append(Hit(c.text, c.metadata, scores[i], rank=rank, origin={"hybrid": c.rank}))
        return out

    # -- comparaison des 3 stratégies -----------------------------------------------------------
    def compare(
        self,
        query: str,
        k: int = 5,
        pool: int | None = None,
        bm25_query: str | None = None,
        sources: list[str] | None = None,
    ) -> dict[str, SearchResult]:
        pool = max(pool or SETTINGS.candidate_pool, k)

        t0 = time.perf_counter()
        dense = self.dense(query, k, sources)
        t_dense = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        candidates = self.hybrid(query, pool, pool, bm25_query, sources)
        t_hybrid = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        reranked = self.rerank(query, candidates, k)
        t_rerank = (time.perf_counter() - t0) * 1000 + t_hybrid  # le reranking inclut la récupération

        hybrid_top = [Hit(h.text, h.metadata, h.score, h.rank, h.origin) for h in candidates[:k]]
        return {
            "dense": SearchResult("dense", dense, t_dense),
            "hybrid": SearchResult("hybrid", hybrid_top, t_hybrid),
            "rerank": SearchResult("rerank", reranked, t_rerank),
        }