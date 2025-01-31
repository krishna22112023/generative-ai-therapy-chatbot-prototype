import json
import os
import logging

logger = logging.getLogger(__name__)


def clean_url(url):
    parts = url.split("#", 1)
    if len(parts) > 1:
        return parts[0], True
    return url, False


# Define file paths
input_file = os.path.join("data", "processed", "processed_NS.json")
output_file = os.path.join("data", "processed", "processed_NS_v2.json")

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Open and read the original JSON file
with open(input_file, "r") as file:
    data = json.load(file)

cleaned_count = 0
total_urls = 0

# Process each record
for record in data:
    if "url" in record:
        total_urls += 1
        record["url"], was_cleaned = clean_url(record["url"])
        if was_cleaned:
            cleaned_count += 1

# Write the modified data to a new file
with open(output_file, "w") as file:
    json.dump(data, file, indent=4)

logger.info(f"URLs have been processed and saved to '{output_file}'.")
logger.info(f"Total URLs processed: {total_urls}")
logger.info(f"URLs cleaned (# removed): {cleaned_count}")
