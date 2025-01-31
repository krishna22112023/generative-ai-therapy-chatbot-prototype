import json
import os
import hashlib
import logging

logger = logging.getLogger(__name__)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def deduplicate_records(data):
    seen_urls = {}
    unique_records = []
    duplicates_removed = 0

    for i, record in enumerate(data):
        url = record.get("url", "")
        text = record.get("text", "")

        if url in seen_urls:
            if text == data[seen_urls[url]].get("text", ""):
                duplicates_removed += 1
                continue

        seen_urls[url] = i
        unique_records.append(record)

    return unique_records, duplicates_removed


# File paths
input_file = os.path.join("data", "processed", "processed_NS_v2.json")
output_file = os.path.join("data", "processed", "processed_NS_v2_dedup.json")

# Load the JSON data
data = load_json(input_file)

# Deduplicate the records
deduplicated_data, removed_count = deduplicate_records(data)

# Save the deduplicated data
save_json(deduplicated_data, output_file)

# Print results
logger.info(f"Original record count: {len(data)}")
logger.info(f"Deduplicated url + text record count: {len(deduplicated_data)}")
logger.info(f"Duplicates removed: {removed_count}")
logger.info(f"Deduplicated data saved to: {output_file}")


def hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def deduplicate_records2(data):
    seen_text_hashes = {}
    unique_records = []
    duplicates_removed = 0

    for record in data:
        text = record.get("text", "")
        text_hash = hash_text(text)

        if text_hash not in seen_text_hashes:
            seen_text_hashes[text_hash] = True
            unique_records.append(record)
        else:
            duplicates_removed += 1

    return unique_records, duplicates_removed


# File paths
input_file = os.path.join("data", "processed", "processed_NS_v2_dedup.json")
output_file = os.path.join("data", "processed", "processed_NS_v3_dedup.json")

# Load the JSON data
data = load_json(input_file)

# Deduplicate the records
deduplicated_data, removed_count = deduplicate_records2(data)

# Save the deduplicated data
save_json(deduplicated_data, output_file)

# Print results
logger.info(f"Original record count: {len(data)}")
logger.info(f"Deduplicated text record count: {len(deduplicated_data)}")
logger.info(f"Duplicates removed: {removed_count}")
logger.info(f"Deduplicated data saved to: {output_file}")
