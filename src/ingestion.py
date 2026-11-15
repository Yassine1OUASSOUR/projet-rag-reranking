"""Pipeline d'ingestion : PDF -> nettoyage -> chunks -> embeddings -> ChromaDB.

Usage :
    python -m src.ingestion            # ajoute uniquement les PDF nouveaux ou modifiés
    python -m src.ingestion --reset    # reconstruit toute la base
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import SETTINGS
from .embeddings import get_embeddings

log = logging.getLogger("ingestion")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # mots coupés en fin de ligne
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path, digest: str) -> list[Document]:
    pages = PyPDFLoader(str(path)).load()
    docs = []
    for page in pages:
        content = clean_text(page.page_content)
        if content:
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": path.name,
                        "page": int(page.metadata.get("page", 0)) + 1,  # pages à partir de 1
                        "file_hash": digest,
                    },
                )
            )
    return docs


def split(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
        add_start_index=True,
    )
    chunks = [c for c in splitter.split_documents(docs) if len(c.page_content) >= SETTINGS.min_chunk_chars]
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
    return chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="Reconstruit la base depuis zéro")
    args = parser.parse_args()

    pdfs = sorted(SETTINGS.data_dir.glob("*.pdf"))
    if not pdfs:
        log.error("Aucun PDF dans %s/", SETTINGS.data_dir)
        return

    store = Chroma(
        collection_name=SETTINGS.collection,
        embedding_function=get_embeddings(),
        persist_directory=str(SETTINGS.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    if args.reset:
        store.delete_collection()
        store = Chroma(
            collection_name=SETTINGS.collection,
            embedding_function=get_embeddings(),
            persist_directory=str(SETTINGS.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    known = {m.get("file_hash") for m in store.get(include=["metadatas"])["metadatas"]}

    for pdf in pdfs:
        digest = file_hash(pdf)
        if digest in known:
            log.info("= %s déjà indexé, ignoré", pdf.name)
            continue

        stale = store.get(where={"source": pdf.name})["ids"]  # version précédente du fichier
        if stale:
            store.delete(ids=stale)
            log.info("~ %s modifié : %d anciens chunks supprimés", pdf.name, len(stale))

        chunks = split(load_pdf(pdf, digest))
        ids = [f"{digest[:12]}-{c.metadata['chunk']}" for c in chunks]
        for start in range(0, len(chunks), 64):
            store.add_documents(chunks[start:start + 64], ids=ids[start:start + 64])
        log.info("+ %s : %d chunks indexés", pdf.name, len(chunks))

    log.info("Terminé : %d chunks au total.", len(store.get()["ids"]))


if __name__ == "__main__":
    main()