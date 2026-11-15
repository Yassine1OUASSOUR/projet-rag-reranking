"""Évaluation chiffrée des 3 stratégies sur un jeu de questions annotées.

Usage : python -m src.evaluation --k 5
Format de eval/questions.json : question, [question_en], [source], [pages], keywords
Un extrait est pertinent s'il vient du bon fichier ET (page attendue OU mot-clé présent).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .retrievers import Hit, RAGComparator


def _norm(s: str) -> str:
    """Ignore casse, espaces, retours à la ligne et tirets (coupures de mots dans les PDF)."""
    return re.sub(r"\W+", "", s.lower())


def is_relevant(hit: Hit, item: dict) -> bool:
    if item.get("source") and hit.source != item["source"]:
        return False
    if hit.page in item.get("pages", []):
        return True
    text = _norm(hit.text)
    return any(_norm(kw) in text for kw in item.get("keywords", []))


def metrics(flags: list[bool]) -> dict[str, float]:
    first = next((i for i, f in enumerate(flags, 1) if f), None)
    return {
        "Hit@k": float(any(flags)),
        "MRR": 1.0 / first if first else 0.0,
        "Precision@k": sum(flags) / max(len(flags), 1),
    }


def run(questions_path: Path, k: int, pool: int) -> pd.DataFrame:
    items = json.loads(questions_path.read_text(encoding="utf-8"))
    comparator = RAGComparator()
    rows = []
    for item in items:
        results = comparator.compare(
            item["question"], k=k, pool=pool, bm25_query=item.get("question_en"),
            sources=[item["source"]] if item.get("source") else None,
        )
        for name, res in results.items():
            rows.append({"strategy": name, "latency_ms": res.latency_ms,
                         **metrics([is_relevant(h, item) for h in res.hits])})
    df = pd.DataFrame(rows).groupby("strategy").mean().round(3)
    return df.loc[["dense", "hybrid", "rerank"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool", type=int, default=30)
    args = parser.parse_args()

    df = run(args.questions, args.k, args.pool)
    df.columns = [c.replace("@k", f"@{args.k}") for c in df.columns]
    out = Path("eval/results.csv")
    df.to_csv(out)
    print(df.to_string())
    print(f"\nRésultats enregistrés dans {out}")


if __name__ == "__main__":
    main()