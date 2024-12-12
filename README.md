

***History of Massachusetts - A RAG System Q&A Bot***

### Overview

This project is a domain-specific Retrieval-Augmented Generation (RAG) application designed to answer questions about the history of Massachusetts. By leveraging a fine-tuned large language model and a vector database, the system provides accurate and contextually rich responses.

### Features

- Semantic search for historical documents.
- Contextually accurate Q&A generation.
- Interactive user interface built with Streamlit.

### Setup Instructions

#### Prerequisites:

- Python 3.8+
- Libraries: Install from `requirements.txt`
- Milvus for vector database setup

#### Steps:

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Ensure the vector database is running and populated with embeddings:
   ```bash
   python scripts/populate_db.py
   ```

### Usage

- Open the Streamlit app in your browser.
- Enter a query related to Massachusetts history.
- View the generated response with contextual details.


### Using the Application

1. Access the Streamlit-hosted UI via the provided URL.
2. Enter your query in plain English (e.g., "What is the significance of the Boston Tea Party?").
3. View the generated response along with relevant historical references.

### Troubleshooting

- If the bot is unresponsive, ensure the vector database is running.
- For incorrect results, verify the query's specificity.



