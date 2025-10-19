import os
import logging

def setup_logging(log_file='app.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_model(model_name):
    from transformers import AutoModel, AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}")
        raise

def save_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)
    logging.info(f"Data saved to {file_path}")

def load_from_csv(file_path):
    import pandas as pd
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        logging.error(f"CSV file not found: {file_path}")
        raise FileNotFoundError(f"CSV file not found: {file_path}")