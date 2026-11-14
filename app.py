import streamlit as st
from src.retrievers import RAGComparator

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Comparateur RAG - Projet RGPD", page_icon="🔍", layout="wide"
)

# Initialisation du comparateur (mis en cache pour ne charger les modèles qu'une seule fois)


@st.cache_resource
def load_comparator():
    return RAGComparator()


with st.spinner(
    "Chargement des modèles et de la base vectorielle en cours..."
):
    comparator = load_comparator()

# Titre de l'application
st.title("⚖️ Comparateur de Stratégies RAG (RGPD)")
st.markdown(
    "Comparez côte à côte l'efficacité de la recherche **Dense (ChromaDB)**, de la recherche **Hybride (BM25 + Dense)** et du **Reranking (Cross-Encoder)**."
)

# Barre latérale pour les réglages
st.sidebar.header("Paramètres de recherche")
strategy_choice = st.sidebar.selectbox(
    "Mode d'affichage",
    [
        "Comparer les 3 stratégies",
        "Dense seul",
        "Hybride",
        "Reranking (Cross-Encoder)",
    ],
)
top_k = st.sidebar.slider(
    "Nombre de résultats à afficher (k)", min_value=1, max_value=10, value=3
)

# Zone de saisie de la question
query = st.text_input(
    "Posez votre question sur le RGPD :",
    value="Quelles sont les sanctions en cas de violation des données ?",
)

if st.button("Lancer la recherche", type="primary"):
    if not query.strip():
        st.warning("Veuillez saisir une question valide.")
    else:
        if strategy_choice == "Comparer les 3 stratégies":
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("1. 🔵 Dense (ChromaDB)")
                results_dense = comparator.dense_search(query, k=top_k)
                for i, r in enumerate(results_dense, 1):
                    st.info(
                        f"**Résultat {i}**\n\n{r['text']}\n\n*Source: Page {r['source'].get('page', 'N/A')}*"
                    )

            with col2:
                st.subheader("2. 🟢 Hybride (BM25 + Dense)")
                results_hybrid = comparator.hybrid_search(query, k=top_k)
                for i, r in enumerate(results_hybrid[:top_k], 1):
                    st.success(
                        f"**Résultat {i}**\n\n{r['text']}\n\n*Source: Page {r['source'].get('page', 'N/A')}*"
                    )

            with col3:
                st.subheader("3. 🟣 Reranking (Cross-Encoder)")
                results_rerank = comparator.rerank_search(
                    query, top_k=top_k
                )
                for i, r in enumerate(results_rerank, 1):
                    score = r.get("score", 0)
                    st.warning(
                        f"**Résultat {i}** (Score: `{score:.4f}`)\n\n{r['text']}\n\n*Source: Page {r['source'].get('page', 'N/A')}*"
                    )
        else:
            # Mode affichage unique
            st.subheader(f"Résultats pour la stratégie : {strategy_choice}")
            if strategy_choice == "Dense seul":
                results = comparator.dense_search(query, k=top_k)
            elif strategy_choice == "Hybride":
                results = comparator.hybrid_search(query, k=top_k)[:top_k]
            else:
                results = comparator.rerank_search(query, top_k=top_k)

            for i, r in enumerate(results, 1):
                score_info = (
                    f" (Score Cross-Encoder: `{r.get('score', 0):.4f}`)"
                    if "score" in r
                    else ""
                )
                st.markdown(
                    f"### Résultat {i}{score_info}\n> {r['text']}\n*Source: Page {r['source'].get('page', 'N/A')}*"
                )
                st.divider()