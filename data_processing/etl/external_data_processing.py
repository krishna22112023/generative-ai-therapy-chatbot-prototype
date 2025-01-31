import json
import csv
import os
import logging

logger = logging.getLogger(__name__)

def preprocess_data(input_file_path,output_file_path):
    """
    Preprocess the input data by:
    
    1. Adding an index as the "ID" field.
    2. Converting the datatypes of fields.
    3. Cleaning string-based fields by removing extra quotes and escape sequences.

    Parameters
    ----------
    data : list of dict
        A list of dictionaries containing the raw data.

    Returns
    -------
    list of dict
        A list of dictionaries with the preprocessed data.
    """
    data = import_csv_as_dict(input_file_path)

    processed_data = []
    for index, item in enumerate(data):
        # Adding index as ID
        item['ID'] = index

        # Converting datatypes and cleaning string-based fields
        item['ID'] = int(item['ID'])
        item['Name'] = clean_string(item.get('Name', ''))
        item['description'] = clean_string(item.get('description', ''))
        item['response_type'] = clean_string(item.get('response_type', ''))
        item['keywords'] = clean_list_of_strings(item.get('keywords', ''))
        item['cat0'] = clean_string(item.get('cat0', ''))
        item['text'] = clean_string(item.get('text', ''))
        item['verbatim_flag'] = convert_to_bool(item.get('verbatim_flag', 'FALSE'))
        item['url'] = clean_string(item.get('url', ''))
        item['duration'] = clean_string(item.get('duration', ''))

        processed_data.append(item)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4)
    
    # Delete the input CSV file
    try:
        os.remove(input_file_path)
        logger.info(f"Input file {input_file_path} has been deleted.")
    except OSError as e:
        logger.error(f"Error deleting input file {input_file_path}: {e}")

def clean_string(s):
    """
    Remove extra quotes, single quotes, and escape sequences from a string.

    Parameters
    ----------
    s : str
        The input string.

    Returns
    -------
    str
        The cleaned string.
    """
    return s.replace('"', '').replace("'", '').replace('\n', ' ').replace('\t', ' ').strip()

def clean_list_of_strings(s):
    """
    Convert a comma-separated string into a list of cleaned strings.

    Parameters
    ----------
    s : str
        The input string, possibly with comma-separated values.

    Returns
    -------
    list of str
        A list of cleaned strings.
    """
    s = s.replace('[', '').replace(']', '')
    return [clean_string(keyword) for keyword in s.split(',')]

def convert_to_bool(value):
    """
    Convert a string to a boolean value.

    Parameters
    ----------
    value : str
        The input string, typically "TRUE" or "FALSE".

    Returns
    -------
    bool
        The corresponding boolean value.
    """
    return value.upper() == 'TRUE'

def import_csv_as_dict(csv_file_path):
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]
    return data

def find_external_data_files(folder_path):
    """
    Find all CSV files in the specified folder and create a list of their paths.

    Parameters
    ----------
    folder_path : str
        The path to the folder where CSV files are to be searched.

    Returns
    -------
    list of str
        A list containing the paths of the CSV files found.
    """
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files