from typing import Literal, List, Dict, Optional
from supabase import Client,create_client
import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import logging

logger = logging.getLogger(__name__)

def perform_search(
    supabase: Client,
    search_type: Literal["hybrid", "keyword", "vector"],
    query_text: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
    match_count: int = 10,
    vector_weight: float = 0.5,
    metadata_filter: Optional[Dict] = None
) -> List[Dict]:
    """
    Perform a search using the specified search type.

    Parameters:
    -----------
    supabase : Client
        The Supabase client.
    search_type : Literal["hybrid", "keyword", "vector"]
        The type of search to perform.
    query_text : str, optional
        The text query for keyword and hybrid search (required for keyword and hybrid).
    query_embedding : List[float], optional
        The embedding vector for vector and hybrid search (required for vector and hybrid).
    match_count : int, optional
        The number of results to return (default is 10).
    vector_weight : float, optional
        The weight given to vector similarity vs. keyword similarity in hybrid search (default is 0.5).
    metadata_filter : Dict, optional
        A dictionary to filter results based on metadata (default is None).

    Returns:
    --------
    List[Dict]
        A list of dictionaries containing the search results.
    """
    if metadata_filter is None:
        metadata_filter = {}

    if search_type == "hybrid":
        if query_text is None or query_embedding is None:
            raise ValueError("Both query_text and query_embedding are required for hybrid search.")
        function_name = 'hybrid_search'
        params = {
            'query_text': query_text,
            'query_embedding': query_embedding,
            'match_count': match_count,
            'vector_weight': vector_weight,
            'metadata_filter': metadata_filter
        }
    elif search_type == "keyword":
        if query_text is None:
            raise ValueError("query_text is required for keyword search.")
        function_name = 'kw_match_documents'
        params = {
            'query_text': query_text,
            'match_count': match_count
        }
    elif search_type == "vector":
        if query_embedding is None:
            raise ValueError("query_embedding is required for vector search.")
        function_name = 'match_documents'
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count,
            'filter': metadata_filter
        }
    else:
        raise ValueError("Invalid search_type. Must be 'hybrid', 'keyword', or 'vector'.")

    try:
        response = supabase.rpc(function_name, params).execute()
        if hasattr(response, 'data'):
            logger.info(f"{search_type} search performed successfully for the given query!")
            return response.data
        else:
            logger.error(f"Error in {search_type} search:", response.error)
            return []
    except Exception as e:
        logger.error(f"Error performing {search_type} search:", str(e))
        return []
    
def retriever_ns_rag(args,query):
    #create supabase client
    supabase_config = {
    "supabase_url": os.environ.get("SUPABASE_URL"),
    "supabase_key": os.environ.get("SUPABASE_KEY")
    }
    supabase: Client = create_client(supabase_config['supabase_url'], supabase_config['supabase_key'])
    
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

    query_embedding = embeddings.embed_query(query)

    results = perform_search(supabase,
                             search_type=args["search_type"],
                             query_text=query,
                             query_embedding=query_embedding,
                             match_count=args['top_k'],
                             vector_weight=args['search_weight'],
                             metadata_filter=args["metadata_filter_NS"]
                             )
    
    return results