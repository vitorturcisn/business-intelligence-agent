from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parents[1]

KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge_base"
INDEX_DIR = BASE_DIR / "data" / "indexes" / "business_knowledge_faiss"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_documents() -> list[Document]:
    documents = []

    files = sorted(KNOWLEDGE_DIR.glob("*.md"))

    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo Markdown encontrado em: {KNOWLEDGE_DIR}"
        )

    for file_path in files:
        content = file_path.read_text(encoding="utf-8")

        if not content.strip():
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "file_type": "markdown",
                },
            )
        )

        print(f"Documento carregado: {file_path.name}")

    return documents


def build_index() -> None:
    print("\nCarregando documentos...")

    documents = load_documents()

    print(f"\nTotal de documentos: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Nenhum chunk foi gerado a partir dos documentos.")

    print(f"Total de chunks gerados: {len(chunks)}")

    print("\nCarregando modelo de embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("\nConstruindo índice FAISS...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(INDEX_DIR))

    print("\nÍndice FAISS criado e salvo com sucesso!")
    print(f"Local: {INDEX_DIR}")
    print(f"Vetores armazenados: {vector_store.index.ntotal}")
    print(f"Arquivos gerados: index.faiss e index.pkl")


if __name__ == "__main__":
    build_index()

