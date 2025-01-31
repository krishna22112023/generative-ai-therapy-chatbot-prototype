import json
import os
import logging
from datetime import datetime
from core.utils import load_events, extract_save_event
import re
from json.decoder import JSONDecodeError
from conf.config import cfg


class EventManager:
    def __init__(self, username, file_path=cfg.user.events):
        self.username = username
        self.file_path = file_path
        self.user_events = load_events(self.file_path, username)
        self.save_event = None
        logging.basicConfig(level=logging.INFO)

    def process_event(self, response):
        """
        Process an event from a response and save it if it's valid.

        Args:
        response (str): The response containing the event.

        Returns:
        str or None: The saved event if successful and valid, "NA" if invalid, None if no event found.
        """

        event = extract_save_event(response)
        if event is None or event == "NA":
            return None  # or return "NA" if you prefer
        return self.save_user_event(event)

    def save_user_event(self, event):
        """
        Save a user event to the JSON file if it's valid.

        Args:
        event (str): The event string in format "YYYYMMDD:description".

        Returns:
        str or None: The saved event if successful and valid, "NA" if invalid, None if error occurs.
        """
        try:
            # Validate and parse the event string
            if not re.match(r"^\d{8}:", event):
                logging.warning(f"Invalid event format: {event}")
                return "NA"

            date_str, description = event.split(":", 1)

            # Check if the description is less than 10 characters
            if len(description.strip()) < 10:
                logging.info(f"Event description too short: {description}")
                return "NA"

            date_obj = datetime.strptime(date_str, "%Y%m%d")
            formatted_date = date_obj.strftime("%Y-%m-%d")

            # Load existing data
            data = self._load_data()

            # Add or update user's events
            if self.username not in data:
                data[self.username] = {}

            if formatted_date not in data[self.username]:
                data[self.username][formatted_date] = []

            data[self.username][formatted_date].append(description.strip())

            # Save updated data
            self._save_data(data)

            self.user_events = data[self.username]
            return event

        except (ValueError, IOError, JSONDecodeError) as e:
            logging.error(f"Error saving event: {str(e)}")
            return None

    def get_events(self):
        """
        Get all events for the current user.

        Returns:
        dict: The user's events.
        """
        return self.user_events

    def _load_data(self):
        """Load data from the JSON file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_data(self, data):
        """Save data to the JSON file."""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
