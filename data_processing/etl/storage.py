from tqdm import tqdm
from supabase import create_client, Client
import os
from typing import Literal,List,Dict,Optional
import logging

logger = logging.getLogger(__name__)

supabase_config = {
    "supabase_url": os.environ.get("SUPABASE_URL"),
    "supabase_key": os.environ.get("SUPABASE_KEY")
}

def setup_database(search_type: Optional[Literal["keyword", "hybrid", "vector"]] = None):
    
    #create supabase client
    supabase: Client = create_client(supabase_config['supabase_url'], supabase_config['supabase_key'])

    functions_to_check = {
        'keyword': 'kw_match_documents',
        'hybrid': 'hybrid_search',
        'vector': 'match_documents'
    }

    if search_type:
        functions_to_check = {search_type: functions_to_check[search_type]}

    results = {}

    for current_type, function_name in functions_to_check.items():
        try:
            # We'll call each function with minimal arguments just to check if it exists
            if current_type == 'keyword':
                response = supabase.rpc(function_name, {'query_text': '', 'match_count': 1}).execute()
            elif current_type == 'hybrid':
                response = supabase.rpc(function_name, {
                    'query_text': '',
                    'query_embedding': [0] * 1536,
                    'match_count': 1
                }).execute()
            elif current_type == 'vector':
                response = supabase.rpc(function_name, {
                    'query_embedding': [0] * 1536,
                    'match_count': 1
                }).execute()

            results[current_type] = True
            logger.info(f"{current_type.capitalize()} search function '{function_name}' exists.")
        except Exception as e:
            results[current_type] = False
            logger.error(f"{current_type.capitalize()} search function '{function_name}' does not exist or is not accessible: {str(e)}")
    if all(results.values()):
        logger.info("All search functions are set up correctly.")
        return supabase
    else:
        logger.error("Some search functions are missing or inaccessible. Please set them up using the Supabase SQL editor.")
        for search_type, status in results.items():
            if not status:
                logger.error(f"- {search_type.capitalize()} search function needs to be set up.")

def load_embeddings_to_database(supabase:Client, embedded_chunks:List[Dict]):
    """Load the embedded chunks to a vector DB

    Args:
        supabase (Client): _description_
        embedded_chunks (List[Dict]): A list of dictionaries, each containing the chunk content,
                    its embedding, and metadata.
    """
    try:
        # Insert the embedded chunks into the database
        for chunk in tqdm(embedded_chunks,desc="loading embedded chunks to vector table"):
            supabase.table('documents').insert({
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'embedding': chunk['embedding']
            }).execute()

        logger.info(f"Successfully initialized Supabase store and inserted {len(embedded_chunks)} chunks.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")