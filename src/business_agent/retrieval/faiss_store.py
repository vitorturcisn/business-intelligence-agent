from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_INDEX_DIR = (
    BASE_DIR / "data" / "indexes" / "business_knowledge_faiss"
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def load_faiss_index(index_dir: str = str(DEFAULT_INDEX_DIR)) -> FAISS:
    """
    Carrega o índice FAISS persistido em disco.

    A desserialização é permitida porque o índice foi gerado
    localmente por este próprio projeto. Não carregar índices
    recebidos de fontes desconhecidas.
    """
    path = Path(index_dir)

    required_files = ["index.faiss", "index.pkl"]

    for filename in required_files:
        if not (path / filename).is_file():
            raise FileNotFoundError(
                f"Arquivo {filename} não encontrado em {path}. "
                "Execute scripts/build_faiss_index.py primeiro."
            )

    embeddings = build_embeddings()

    return FAISS.load_local(
        folder_path=str(path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
