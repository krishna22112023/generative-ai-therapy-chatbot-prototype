import json
import os
from jsonschema import validate
from langchain.schema.document import Document
import logging

logger = logging.getLogger(__name__)

def load_documents(folder_path, schema_path=None):
    """
    Load documents from a JSON file and optionally validate against a schema.

    This function reads a JSON file containing document data, optionally validates it
    against a provided schema, and creates a list of Document objects.

    Args:
        file_path (str): Path to the JSON file containing document data.
        schema_path (str, optional): Path to the JSON schema file. If provided,
                                     the input data will be validated against this schema.

    Returns:
        list: A list of Document objects created from the JSON data.

    Raises:
        FileNotFoundError: If the specified file_path or schema_path is not found.
        json.JSONDecodeError: If the JSON in file_path or schema_path is invalid.
        jsonschema.exceptions.ValidationError: If the data doesn't match the provided schema.
        KeyError: If a required key is missing in the JSON data.

    Example:
        >>> documents = load_documents('data.json', 'schema.json')
        >>> logger.info(len(documents))
        5
    """
    try:
        #combine all the json files
        processed_data = combine_json_files(folder_path) 

        # Load and validate schema if provided
        '''if schema_path:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            validate(instance=processed_data, schema=schema)'''

        documents = []
        for item in processed_data:
            # Dynamically create content based on available fields
            content_fields = ['Name', 'description', 'text']
            content = '\n'.join(item.get(field, '') for field in content_fields if field in item)

            # Dynamically create metadata
            metadata = {key: value for key, value in item.items() if key != 'text'}

            # Create a Document object
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        logger.info("successfully loaded json to langchain document!")
        return documents

    except FileNotFoundError as e:
        logger.error(f"Error: File not found - {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Error: Invalid JSON format - {e}")
    except KeyError as e:
        logger.error(f"Error: Missing key in JSON data - {e}")
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred - {e}")

    return []

def combine_json_files(folder_path):
    combined_data = []

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            
            # Open and read the JSON file
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    combined_data += data
                except json.JSONDecodeError as e:
                    logger.error(f"Error reading {filename}: {e}")
    
    return combined_data