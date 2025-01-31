"""Module for processing documents and creating nodes for use in the RAG model."""

from typing import List,Dict
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.docstore.document import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import logging

logger = logging.getLogger(__name__)

def process_documents(documents: List[Document], args: Dict) -> List[Document]:
    """
    Process documents and create nodes for the RAG model based on specified parameters.

    Args:
        documents (List[Document]): A list of Document objects to be processed.
        args (Dict): A dictionary containing processing parameters:
            - chunk_size (int): The size of each chunk.
            - chunk_overlap (int): The overlap between chunks.
            - retrieval_method (str): The method used for retrieval.
            - chunking_method (str): The method used for chunking.

    Returns:
        List[Document]: A list of processed Document objects (nodes).

    Raises:
        NotImplementedError: If the specified retrieval method is not implemented.
    """
    # Get chunking parameters and retrieval method
    chunk_size = args["chunk_size"]
    chunk_overlap = args["chunk_overlap"]
    retrieval_method = args["retrieval_method"]
    chunking_method = args["chunking_method"]

    # Apply indexing method
    if retrieval_method == "naive":
        nodes = convert_documents_to_chunks(
            documents=documents,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise NotImplementedError(f"Retrieval method '{retrieval_method}' not implemented")

    return nodes

def convert_documents_to_chunks(documents: List[Document], chunking_method: str = "RecursiveCharacterTextSplitter", **kwargs) -> List[Document]:
    """
    Convert documents to chunks using the specified chunking method.

    Args:
        documents (List[Document]): A list of Document objects to be chunked.
        chunking_method (str): The method to use for chunking. 
            Options: "RecursiveCharacterTextSplitter", "SemanticChunker".
        **kwargs: Additional keyword arguments for the specific chunking method.

    Returns:
        List[Document]: A list of chunked Document objects.

    Raises:
        NotImplementedError: If the specified chunking method is not implemented.

    Example:
    # For RecursiveCharacterTextSplitter
    chunks = convert_documents_to_chunks(
        documents, 
        chunking_method="RecursiveCharacterTextSplitter", 
        chunk_size=1000, 
        chunk_overlap=200
    )

    # For SemanticChunker
    chunks = convert_documents_to_chunks(
        documents, 
        chunking_method="SemanticChunker", 
        breakpoint_threshold_type="standard_deviation"
    )
    """
    chunking_methods = {
        "RecursiveCharacterTextSplitter": lambda: RecursiveCharacterTextSplitter(**kwargs),
        "SemanticChunker": lambda: SemanticChunker(OpenAIEmbeddings(), **kwargs)
    }

    if chunking_method not in chunking_methods:
        raise NotImplementedError(f"Chunking method '{chunking_method}' not implemented")

    text_splitter = chunking_methods[chunking_method]()

    if chunking_method == "RecursiveCharacterTextSplitter":
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Successfully converted documents to chunks using {chunking_method}")
        return chunks
    elif chunking_method == "SemanticChunker":
        chunks = text_splitter.create_documents(documents)
        logger.info(f"Successfully converted documents to chunks using {chunking_method}")
        return chunks

def convert_chunks_to_embeddings(args: Dict, chunks: List[Document]) -> List[Dict]:
    """
    Convert document chunks to embeddings using the specified embedding model.

    Args:
        args (Dict): A dictionary containing embedding configuration:
        - type (str): The type of embedding model to use.
        - model (str): The specific model to use for embeddings.
        - Additional parameters specific to each embedding type.
        chunks (List[Document]): A list of Document objects to be embedded.

    Returns:
        List[Dict]: A list of dictionaries, each containing the chunk content,
                    its embedding, and metadata.

    Raises:
        ValueError: If an unsupported embedding type is specified.
    """
    embedding_config = args.get("embedding", {})
    embedding_type = embedding_config.get("type")
    model = embedding_config.get("model")

    if embedding_type == "OpenAIEmbeddings":
        embeddings = OpenAIEmbeddings(model=model)
    elif embedding_type == "HuggingFaceEmbeddings":
        embed_batch_size = embedding_config.get("embed_batch_size", 32)
        embeddings = HuggingFaceEmbeddings(model_name=model, embed_batch_size=embed_batch_size)
    elif embedding_type == "HuggingFaceInferenceAPIEmbeddings":
        base_url = embedding_config.get("base_url")
        embeddings = HuggingFaceInferenceAPIEmbeddings(model=model, base_url=base_url)
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    embedded_chunks = []
    for chunk in tqdm(chunks, desc="Creating embeddings for individual chunks"):
        embedding = embeddings.embed_query(chunk.page_content)
        embedded_chunks.append({
            "content": chunk.page_content,
            "embedding": embedding,
            "metadata": chunk.metadata
        })

    return embedded_chunks