# Embed MedSigLip Project

## Overview
The Embed MedSigLip project is designed to facilitate the embedding of medical images and texts using the MedSigCLIP model. This project provides an API for users to interact with the embedding functionalities, allowing for efficient retrieval and processing of medical data.

## Project Structure
```
Embed_MedSigLip
├── src
│   ├── api.py                # Entry point for the API using Flask or FastAPI
│   ├── fetch_ehr.py          # Functions to fetch electronic health records (EHR) data
│   ├── hybrid_retrieval.py    # Implements hybrid retrieval methods for image and text embeddings
│   ├── medsiglip_embeddings.py # Logic for embedding images and texts using MedSigCLIP
│   └── utils.py              # Utility functions for data preprocessing and file handling
├── requirements.txt           # Lists project dependencies
└── README.md                  # Documentation for the project
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd MedSigLip
pip install -r requirements.txt
```

## Usage
To run the API, execute the following command:

```bash
python src/api.py
```

This will start the web server, and you can access the API endpoints for embedding images and texts.

## API Endpoints
- **POST /embed**: Accepts an image and text input, returns the corresponding embeddings.
- **GET /retrieve**: Queries the FAISS index for relevant embeddings based on input parameters.

## Model Download
The MedSigCLIP model will be automatically downloaded from Hugging Face if it is not available locally. Ensure you have an internet connection when running the embedding functions for the first time.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.