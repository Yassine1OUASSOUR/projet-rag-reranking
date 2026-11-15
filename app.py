"""Dashboard Streamlit : comparaison de stratégies RAG et génération locale via Ollama."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import SETTINGS
from src.llm import OllamaClient, build_messages
from src.retrievers import Hit, RAGComparator, SearchResult

st.set_page_config(page_title="RAG Lab", page_icon="🔬", layout="wide")

STRATEGIES = {
    "dense": ("Dense", "#2563EB", "Similarité sémantique (embeddings E5)"),
    "hybrid": ("Hybride", "#0F766E", "Dense + BM25 fusionnés par RRF"),
    "rerank": ("Reranking", "#7C3AED", "Hybride + cross-encoder BGE"),
}

st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
.pill {display:inline-block; padding:1px 9px; border-radius:6px; font-size:.78rem; font-weight:600; margin:0 6px 6px 0;}
.pill-muted {background:rgba(128,128,128,.16);}
.hit-text {font-size:.92rem; line-height:1.5;}
div[data-testid="stMetric"] {border-left:3px solid rgba(128,128,128,.35); padding-left:.8rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------------------------
# Ressources
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement des modèles et de l'index…")
def load_comparator() -> RAGComparator:
    return RAGComparator()


@st.cache_resource
def load_llm() -> OllamaClient:
    return OllamaClient()


try:
    comparator = load_comparator()
except RuntimeError as exc:
    st.error(f"{exc}")
    st.stop()
llm = load_llm()


def pill(text: str, color: str | None = None) -> str:
    style = f"background:{color}22;color:{color};" if color else ""
    cls = "pill" if color else "pill pill-muted"
    return f"<span class='{cls}' style='{style}'>{html.escape(text)}</span>"


def score_label(strategy: str, hit: Hit) -> str:
    if strategy == "dense":
        return f"similarité {hit.score * 100:.1f} %"
    if strategy == "rerank":
        return f"pertinence {hit.score * 100:.1f} %"
    return f"RRF {hit.score:.4f}"


def render_hit(strategy: str, hit: Hit, expanded: bool = False) -> None:
    _, color, _ = STRATEGIES[strategy]
    badges = pill(f"#{hit.rank}", color) + pill(f"{hit.source} · p. {hit.page}") + pill(score_label(strategy, hit))
    for name, rank in hit.origin.items():
        badges += pill(f"{name} #{rank}")
    with st.container(border=True):
        st.markdown(badges, unsafe_allow_html=True)
        excerpt = hit.text if expanded else (hit.text[:320] + ("…" if len(hit.text) > 320 else ""))
        st.markdown(f"<div class='hit-text'>{html.escape(excerpt)}</div>", unsafe_allow_html=True)
        if not expanded and len(hit.text) > 320:
            with st.expander("Texte complet"):
                st.write(hit.text)


# ---------------------------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------------------------
installed = llm.models()
with st.sidebar:
    st.header("Paramètres")

    selected_sources = st.multiselect(
        "Documents interrogés", comparator.sources, default=comparator.sources,
        help="Restreint la recherche aux documents choisis.",
    )
    top_k = st.slider("Extraits conservés (k)", 1, 8, 4)
    pool = st.slider("Candidats avant reranking", 10, 50, SETTINGS.candidate_pool, step=5,
                     help="Nombre de passages issus de la fusion RRF évalués par le cross-encoder.")
    context_strategy = st.radio(
        "Stratégie envoyée au LLM", list(STRATEGIES), index=2,
        format_func=lambda s: STRATEGIES[s][0],
    )
    translate = st.toggle(
        "Traduire la requête pour BM25", value=False,
        help="Utile quand la question est en français et le document en anglais.",
    )

    st.divider()
    st.subheader("Génération")
    if installed:
        default = next((i for i, m in enumerate(installed) if m.startswith(SETTINGS.ollama_model)), 0)
        model = st.selectbox("Modèle Ollama", installed, index=default)
    else:
        model = st.text_input("Modèle Ollama", SETTINGS.ollama_model)
    temperature = st.slider("Température", 0.0, 1.0, 0.1, 0.05)
    max_tokens = st.slider("Longueur max. (tokens)", 100, 1500, 600, step=50)

    st.divider()
    st.caption("Services")
    st.markdown(
        pill("Ollama : actif" if installed else "Ollama : injoignable", "#15803D" if installed else "#B91C1C")
        + pill(f"{len(comparator.texts)} chunks")
        + pill(f"calcul : {comparator.device}"),
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------------------------
# En-tête et saisie
# ---------------------------------------------------------------------------------------------
st.title("RAG Lab")
st.caption("Comparez trois stratégies de recherche et générez une réponse sourcée, entièrement en local.")

with st.form("ask", border=False):
    c_in, c_btn = st.columns([6, 1], vertical_alignment="bottom")
    question = c_in.text_input(
        "Votre question", placeholder="Ex. : Quelle est la différence entre RAG-Sequence et RAG-Token ?"
    )
    submitted = c_btn.form_submit_button("Rechercher", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Saisissez une question.")
    elif not selected_sources:
        st.warning("Sélectionnez au moins un document dans la barre latérale.")
    else:
        bm25_query = None
        if translate and installed:
            with st.spinner("Traduction de la requête…"):
                bm25_query = llm.translate_to_english(question, model)
        with st.spinner("Recherche en cours…"):
            results = comparator.compare(
                question, k=top_k, pool=pool, bm25_query=bm25_query,
                sources=selected_sources if len(selected_sources) < len(comparator.sources) else None,
            )
        st.session_state["run"] = {
            "question": question, "results": results, "bm25_query": bm25_query,
            "answer": None, "stats": None, "context": context_strategy, "model": model,
        }

run = st.session_state.get("run")
tab_answer, tab_compare, tab_corpus = st.tabs(["Réponse", "Comparaison", "Corpus et évaluation"])

# ---------------------------------------------------------------------------------------------
# Onglet Réponse
# ---------------------------------------------------------------------------------------------
with tab_answer:
    if not run:
        st.info("Posez une question pour lancer la recherche et la génération.")
    else:
        results: dict[str, SearchResult] = run["results"]
        used = results[run["context"]]
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.subheader("Réponse")
            if run["bm25_query"]:
                st.caption(f"Requête BM25 traduite : {run['bm25_query']}")
            box = st.empty()
            if run["answer"] is None:
                if not used.hits:
                    run["answer"] = "Aucun extrait trouvé pour cette question."
                    box.warning(run["answer"])
                else:
                    text = ""
                    try:
                        for chunk in llm.chat_stream(
                            run["model"], build_messages(run["question"], used.hits), temperature, max_tokens
                        ):
                            text += chunk.get("token", "")
                            box.markdown(text + "▌")
                            if chunk.get("done"):
                                run["stats"] = chunk
                        run["answer"] = text
                        box.markdown(text)
                    except Exception as exc:  # réseau, modèle absent, timeout…
                        box.error(f"Échec de la génération avec Ollama ({run['model']}) : {exc}")
            else:
                box.markdown(run["answer"])

            if run["stats"]:
                m1, m2, m3 = st.columns(3)
                m1.metric("Tokens générés", run["stats"]["tokens"])
                m2.metric("Vitesse", f"{run['stats']['tps']:.1f} tok/s")
                m3.metric("Modèle", run["model"])

        with right:
            st.subheader(f"Sources ({STRATEGIES[run['context']][0]})")
            for hit in used.hits:
                render_hit(run["context"], hit)

# ---------------------------------------------------------------------------------------------
# Onglet Comparaison
# ---------------------------------------------------------------------------------------------
with tab_compare:
    if not run:
        st.info("Lancez une recherche pour comparer les stratégies.")
    else:
        results = run["results"]
        ref_keys = {h.key for h in results["rerank"].hits}

        summary = pd.DataFrame(
            [
                {
                    "Stratégie": STRATEGIES[name][0],
                    "Latence (ms)": round(res.latency_ms),
                    "Extraits": len(res.hits),
                    "Recouvrement avec reranking": (
                        f"{len({h.key for h in res.hits} & ref_keys)}/{len(ref_keys)}"
                    ),
                }
                for name, res in results.items()
            ]
        )
        col_tbl, col_chart = st.columns([3, 2], gap="large")
        col_tbl.dataframe(summary, hide_index=True, use_container_width=True)
        col_chart.bar_chart(summary.set_index("Stratégie")["Latence (ms)"], height=180)
        st.caption("La latence du reranking inclut la recherche hybride qui le précède.")

        cols = st.columns(3, gap="medium")
        for col, (name, res) in zip(cols, results.items()):
            label, color, desc = STRATEGIES[name]
            with col:
                st.markdown(f"#### {label}")
                st.caption(f"{desc} · {res.latency_ms:.0f} ms")
                if not res.hits:
                    st.warning("Aucun résultat.")
                for hit in res.hits:
                    render_hit(name, hit)

# ---------------------------------------------------------------------------------------------
# Onglet Corpus et évaluation
# ---------------------------------------------------------------------------------------------
with tab_corpus:
    st.subheader("Corpus indexé")
    corpus = pd.DataFrame(comparator.corpus_summary())
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(corpus))
    c2.metric("Chunks", int(corpus["Chunks"].sum()))
    c3.metric("Taille de chunk", f"{SETTINGS.chunk_size} / {SETTINGS.chunk_overlap}")
    st.dataframe(corpus, hide_index=True, use_container_width=True)

    st.subheader("Évaluation des stratégies")
    results_path = Path("eval/results.csv")
    if results_path.exists():
        df = pd.read_csv(results_path, index_col=0)
        st.dataframe(df, use_container_width=True)
        metric_cols = [c for c in df.columns if c.startswith(("Hit", "MRR", "Precision"))]
        st.bar_chart(df[metric_cols])
    else:
        st.info("Aucune évaluation disponible. Complétez `eval/questions.json` puis lancez `python -m src.evaluation`.")