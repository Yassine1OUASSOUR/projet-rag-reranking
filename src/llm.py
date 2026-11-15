"""Client Ollama (streaming) et construction du prompt ancré dans les sources."""
from __future__ import annotations

import json
from typing import Iterator

import requests

from .config import SETTINGS
from .retrievers import Hit

SYSTEM_PROMPT = (
    "Tu es un assistant de recherche documentaire. Réponds uniquement à partir des extraits numérotés fournis. "
    "Les extraits peuvent être en anglais ; rédige toujours ta réponse dans la langue de la question. "
    "Cite tes sources avec leur numéro entre crochets, par exemple [1] ou [2]. "
    "Si les extraits ne contiennent pas l'information, dis-le clairement au lieu d'inventer."
)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context = "\n\n".join(f"[{i}] ({h.source}, p. {h.page})\n{h.text}" for i, h in enumerate(hits, 1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extraits :\n{context}\n\nQuestion : {question}"},
    ]


class OllamaClient:
    def __init__(self, base_url: str = SETTINGS.ollama_url, timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def models(self) -> list[str]:
        """Liste des modèles installés ; liste vide si le serveur est injoignable."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            return sorted(m["name"] for m in r.json().get("models", []))
        except requests.RequestException:
            return []

    def chat_stream(
        self, model: str, messages: list[dict], temperature: float = 0.1, num_predict: int = 600
    ) -> Iterator[dict]:
        """Produit {'token': str} puis {'done': True, 'tokens': int, 'tps': float}."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": 4096},
        }
        with requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if token := data.get("message", {}).get("content"):
                    yield {"token": token}
                if data.get("done"):
                    n, dur = data.get("eval_count", 0), data.get("eval_duration", 0)
                    yield {"done": True, "tokens": n, "tps": n / (dur / 1e9) if dur else 0.0}

    def translate_to_english(self, text: str, model: str) -> str:
        """Traduit la requête pour que BM25 puisse matcher un corpus anglais."""
        messages = [
            {"role": "system", "content": "Translate the user's question to English. Output only the translation."},
            {"role": "user", "content": text},
        ]
        out = "".join(c.get("token", "") for c in self.chat_stream(model, messages, temperature=0, num_predict=80))
        return out.strip() or text