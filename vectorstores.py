from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.docstore.document import Document
import os
from langchain.agents.agent_toolkits import (
    VectorStoreToolkit,
    VectorStoreInfo,
)
import json


def load_directory(dir):
    loaders = []

    for root, dirs, files in os.walk(os.path.join("vectorstores", dir)):
        for filename in files:
            if filename == "vectorstore_metadata.json":
                continue

            if filename.endswith(".txt"):
                loaders.append(
                    TextLoader(
                        os.path.join(root, filename),
                    )
                )
            elif filename.endswith(".md"):
                loaders.append(
                    UnstructuredMarkdownLoader(
                        os.path.join(root, filename),
                    )
                )

    return loaders


def load_memory_vectorstore(dir, metadata):
    all_lines = []

    # for each file in dir, split it into lines and append to a list
    for filename in os.listdir(os.path.join("vectorstores", dir)):
        if filename == "vectorstore_metadata.json":
            continue

        if filename.endswith(".txt"):
            with open(os.path.join("vectorstores", dir, filename), "r") as f:
                lines = f.readlines()
                all_lines.extend(lines)

    docs = [Document(page_content=line) for line in all_lines]

    return split_and_make_vectorstore(docs)


def split_and_make_vectorstore(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)

    embeddings = SentenceTransformerEmbeddings()

    return Chroma.from_documents(documents, embeddings)


def load_vectorstore(dir, metadata):
    docs = []

    loaders = load_directory(dir)

    for loader in loaders:
        docs.extend(loader.load())

    vectorstore = split_and_make_vectorstore(docs)

    return VectorStoreInfo(
        name=metadata["name"],
        description=metadata["description"],
        vectorstore=vectorstore,
    )


def load_vectorstores():
    vectorstores = []
    memory_vectorstore = None

    for item in os.listdir("vectorstores"):
        if os.path.isdir(os.path.join("vectorstores", item)):
            if not os.path.exists(
                os.path.join("vectorstores", item, "vectorstore_metadata.json")
            ):
                continue

            metadata = json.load(
                open(os.path.join("vectorstores", item, "vectorstore_metadata.json"))
            )

            vectorstore = load_vectorstore(item, metadata)

            if metadata["name"] == "memory":
                memory_vectorstore = load_memory_vectorstore(item, metadata)

            vectorstores.append(vectorstore)

    return memory_vectorstore, vectorstores
