import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIRECTORY = "./chroma_db_ollama"
SOURCE_FILE = "data/my_knowledge.txt"


def _database_exists() -> bool:
    return os.path.exists(os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3"))


def init_database(force: bool = False) -> bool:
    if not force and _database_exists():
        print("知识库已存在，跳过初始化。")
        return False

    if not os.path.exists(SOURCE_FILE):
        print(f"找不到知识库源文件: {SOURCE_FILE}")
        return False

    print("开始构建本地知识库...")
    loader = TextLoader(SOURCE_FILE, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    print("知识库构建完成。")
    return True


if __name__ == "__main__":
    init_database(force=True)
