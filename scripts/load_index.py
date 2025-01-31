import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import data_processing as dp
import logging

logger = logging.getLogger(__name__)

def main(args) -> None:
    try:
        #look for externally uploaded csv, preprocess them, and save in processed folder path
        csv_files = dp.etl.external_data_processing.find_external_data_files(args["external_upload_path"])
        if len(csv_files)>0:
            for file in csv_files:  
                dp.etl.external_data_processing.preprocess_data(file,args["processed_folder"]+os.path.basename(file).split('.csv')[0]+'.json')
        else:
            logger.info("no externally uploaded files found. Continue to load index")

        # Load documents from JSON file
        documents = dp.etl.document_loading.load_documents(args["processed_folder"],args["processed_schema_path"])

        # Process documents into chunks
        chunks = dp.etl.document_processing.process_documents(documents, args)

        # Process chunks into chunk embeddings
        embedded_chunks = dp.etl.document_processing.convert_chunks_to_embeddings(args,chunks)

        # Initialize supabase client for keyword/vector/hybrid search 
        vector_store_client = dp.etl.storage.setup_database(args["search_type"])

        # Load embeddings to database
        dp.etl.storage.load_embeddings_to_database(vector_store_client,embedded_chunks)
        
    except Exception as e:
        logger.error(f"Error in processing: {e}")


if __name__ == "__main__":
    args = {
        "external_upload_path":"./data/external_uploads",
        "processed_folder":"./data/processed/",
        "processed_schema_path":"./data/schema/processed_data_schema.json",
        "chunk_size": 1000,
        "chunk_overlap": 20,
        "chunking_method": "RecursiveCharacterTextSplitter",
        "retrieval_method": "naive",
        "embedding":{"type":"OpenAIEmbeddings","model":"text-embedding-ada-002"},
        "search_type":"hybrid"
        }
    main(args)
