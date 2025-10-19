import os
import requests

def fetch_ehr_data(api_url, params=None):
    """Fetch electronic health records (EHR) data from the specified API."""
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()  # Return the JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching EHR data: {e}")
        return None

def save_ehr_data_to_file(data, file_path):
    """Save fetched EHR data to a specified file."""
    try:
        with open(file_path, 'w') as file:
            file.write(data)
        print(f"EHR data saved to {file_path}")
    except IOError as e:
        print(f"Error saving EHR data to file: {e}")

def load_ehr_data_from_file(file_path):
    """Load EHR data from a specified file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except IOError as e:
        print(f"Error loading EHR data from file: {e}")
        return None