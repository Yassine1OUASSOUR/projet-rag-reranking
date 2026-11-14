import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print(f"Recherche de PDF dans : {DATA_DIR.absolute()}")
    
    if not DATA_DIR.exists():
        print(f"Erreur : Le dossier {DATA_DIR} n'existe pas.")
        return

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    print(f"Fichiers PDF trouvés : {[p.name for p in pdf_files]}")

    if not pdf_files:
        print("Aucun PDF trouvé ! Veuillez placer un PDF dans le dossier data/.")
        return

    print("1. Chargement des PDF...")
    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    documents = loader.load()
    print(f"   -> {len(documents)} pages chargées avec succès.")

    print("2. Découpage en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> {len(chunks)} chunks générés.")

    print(f"3. Chargement du modèle d'embedding ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"4. Création de la base ChromaDB dans {CHROMA_DIR}...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print("   -> Indexation terminée ! Le dossier chroma_db/ a été créé.")

if __name__ == "__main__":
    main()