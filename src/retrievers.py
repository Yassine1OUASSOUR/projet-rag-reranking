from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

CHROMA_DIR = Path("chroma_db")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # Reranker multilingue de pointe


class RAGComparator:

    def __init__(self):
        print("Chargement des embeddings et de la base ChromaDB...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            persist_directory=str(CHROMA_DIR), embedding_function=self.embeddings
        )

        # Récupération de tous les documents stockés pour initialiser BM25
        data = self.vector_store.get()
        self.texts = data["documents"]
        self.metadatas = data["metadatas"]

        print("Initialisation de l'index BM25 (recherche lexicale)...")
        tokenized_corpus = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Chargement du modèle de Reranking multilingue (Cross-Encoder)...")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL
        )

    def dense_search(self, query: str, k: int = 5):
        """1. Stratégie Dense (Similarité sémantique via ChromaDB)."""
        docs = self.vector_store.similarity_search(query, k=k)
        return [{"text": doc.page_content, "source": doc.metadata} for doc in docs]

    def hybrid_search(self, query: str, k: int = 10):
        """2. Stratégie Hybride combinant BM25 et Dense via RRF (Reciprocal Rank Fusion)."""
        # Récupération des résultats Denses
        dense_docs = self.vector_store.similarity_search(query, k=k)

        # Récupération des résultats BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:k]

        # Application du RRF (Reciprocal Rank Fusion) pour fusionner proprement les rangs
        rrf_scores = {}
        doc_lookup = {}

        # Notation des rangs denses (constante k_rrf = 60)
        for rank, doc in enumerate(dense_docs):
            text = doc.page_content
            doc_lookup[text] = {"text": text, "source": doc.metadata}
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (60 + (rank + 1))

        # Notation des rangs BM25
        for rank, idx in enumerate(top_bm25_indices):
            text = self.texts[idx]
            if text not in doc_lookup:
                doc_lookup[text] = {
                    "text": text,
                    "source": self.metadatas[idx],
                }
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (60 + (rank + 1))

        # Tri des documents selon leur score RRF combiné décroissant
        sorted_docs = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        return [doc_lookup[text] for text in sorted_docs[:k]]

    def rerank_search(self, query: str, k_initial: int = 20, top_k: int = 3):
        """3. Stratégie Hybride + Reranking (Cross-Encoder multilingue)."""
        candidates = self.hybrid_search(query, k=k_initial)
        if not candidates:
            return []

        pairs = [[query, cand["text"]] for cand in candidates]
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )

        with torch.no_grad():
            logits = self.reranker_model(**inputs).logits.squeeze(-1)
            # Normalisation sigmoïde pour obtenir un score de pertinence entre 0 et 1 (en pourcentage)
            scores = torch.sigmoid(logits)

        ranked_indices = torch.argsort(scores, descending=True).tolist()

        reranked_results = []
        for idx in ranked_indices[:top_k]:
            res = candidates[idx]
            res["score"] = float(scores[idx]) * 100  # Exprimé en %
            reranked_results.append(res)

        return reranked_results