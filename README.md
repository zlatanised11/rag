# History of Massachusetts Q&A Bot

## Overview

The **History of Massachusetts Q&A Bot** is a Retrieval-Augmented Generation (RAG) system designed to answer questions about Massachusetts history. It leverages Wikipedia as the primary knowledge source, employs vector embeddings for semantic search, and integrates OpenAI's GPT-3.5-turbo for natural language responses. The system processes historical data into Markdown and CSV formats and uses a Streamlit interface for user interaction.

## Features

- **Data Collection**: Extracts structured data from Wikipedia using the `wikipedia-api` library.
- **Semantic Search**: Utilizes FAISS for fast and accurate similarity-based document retrieval.
- **Concise Answers**: GPT-3.5-turbo generates precise responses to user queries.
- **Interactive Frontend**: Streamlit provides an accessible interface for user interaction.

## Advantages of the RAG-Based Q&A Bot Over ChatGPT

1. **Custom Knowledge Base**
   - The bot uses Wikipedia as its source, ensuring tailored and up-to-date answers.
   - Unlike ChatGPT, which relies on static training data, this bot's knowledge base can be updated anytime.

2. **Traceable and Verifiable Responses**
   - Each answer is backed by specific documents or sections from the knowledge base, making it easy to verify.
   - ChatGPT does not provide citations or traceable sources for its responses.

3. **Cost Efficiency**
   - By retrieving only relevant chunks of information using a vector store (FAISS), the bot reduces token usage with GPT-3.5-turbo.
   - This optimization lowers API costs compared to directly querying ChatGPT.

4. **Reduced Hallucinations**
   - The RAG bot minimizes inaccuracies by generating responses strictly based on retrieved, factual data.
   - ChatGPT may hallucinate and produce plausible but incorrect information when it lacks sufficient context.

5. **Domain-Specific Expertise**
   - The bot specializes in specific topics (e.g., Massachusetts history) by curating its knowledge base.
   - ChatGPT provides general knowledge, which might lack precision in niche domains.

The RAG-based Q&A Bot is more **accurate**, **cost-effective**, and **reliable** for domain-specific applications compared to directly using ChatGPT.

## File Structure

- **`Wiki_Scraper.py`**: Script for scraping and processing Wikipedia pages. Saves data as CSV and Markdown files. 
- **`RAGChat.py`**: Core implementation of the RAG system, including document chunking, embedding generation, and retrieval logic. 
- **`app.py`**: Streamlit application for hosting the Q&A bot. Handles user queries and displays responses interactively. 
- **`requirements.txt`**: Lists Python dependencies required for the project. 
- **`output.md`**: Example of Markdown output generated by the Wikipedia scraper.
- **`markdown.py`**: Converts CSV files into Markdown format, organizing content by headings and sections for improved readability.

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Setup Steps
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Install Dependencies**:
   Use `pip` to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure OpenAI API**:
   Set your OpenAI API key in the Streamlit secrets file:
   ```bash
   mkdir -p ~/.streamlit
   echo "[secrets]\napi='<YOUR_API_KEY>'" > ~/.streamlit/secrets.toml
   ```
4. **Run the Wiki Scraper Script**:
   Fetch, process, and save Wikipedia content as CSV files
   ```bash
   python Wiki_Scraper.py
   ```
5. **Generate Markdown Files**:
   Convert the CSV files into a single Markdown file for easy sharing or documentation
   ```bash
   python markdown.py
   ```
6. **Run Streamlit**:
   Host the application on your local network
   ```bash
   streamlit run app.py
   ```
   
## Project Tree
   ```
📦 
├─ .devcontainer
│  └─ devcontainer.json
├─ HistoryOfMassProjectReportRAG.pdf
├─ RAGChat.py
├─ README.md
├─ Wiki_Scraper.py
├─ app.py
├─ markdown.py
├─ output.md
└─ requirements.txt
```
©generated by [Project Tree Generator](https://woochanleee.github.io/project-tree-generator)

## Usage

### Data Preparation

This script prepares and processes data from multiple Wikipedia pages into structured formats for further analysis and documentation. The following steps outline the process:

### 1. Setup and Initialization
- **Directory Creation**: A directory named `raw_data` is created if it does not already exist. This directory will store individual CSV files generated for each Wikipedia page.
- **Wikipedia API Initialization**: The `wikipediaapi` library is initialized for the English Wikipedia with a custom user-agent (`RAG 2 Riches`) to ensure smooth interaction with the API.

### 2. Fetching and Processing Wikipedia Data
- **Page Retrieval**: The script uses the `wikipediaapi` library to fetch data for a list of specified Wikipedia pages.
- **Exclusion of Unwanted Sections**: Sections such as `See also`, `References`, `Bibliography`, `External links`, `Explanatory notes`, and `Further reading` are excluded to focus on relevant content.
- **Recursive Section Parsing**: A recursive function processes each section and its subsections to extract meaningful text. Full section titles are generated to maintain context.
- **Saving as CSV**: The extracted data for each page is saved as a CSV file in the `raw_data` folder. Each CSV file includes two columns:
  - `section`: The section title.
  - `text`: The content.

### 3. Combining Data
- **Loading Individual CSV Files**: All CSV files in the `raw_data` folder are loaded into separate pandas DataFrames.
- **Combining DataFrames**: These DataFrames are concatenated into a single DataFrame containing all the sections and text from the processed Wikipedia pages.
- **Saving Combined Data**: The combined DataFrame is saved as a single CSV file named `combined_data.csv`.

### 4. Markdown Conversion
- **CSV to Markdown Table**: The combined CSV file (`combined_data.csv`) is converted into a Markdown table using the `pandas` library. The Markdown table is saved as `combined_data.md`.
- **CSV to Detailed Markdown Document**:
  - Each CSV file is individually processed to create a detailed Markdown file.
  - The Markdown file includes:
    - A main heading for each Wikipedia page (based on the CSV file name).
    - Subheadings for each section title.
    - Corresponding text content under each section.
  - The final Markdown document (`output.md`) is saved in the `raw_data` folder.

### 5. Error Handling
- The script checks if each Wikipedia page exists. If not, an error message is displayed, and the page is skipped.
- During Markdown conversion, the script ensures that required columns (`section` and `text`) exist in each CSV file, displaying warnings for missing columns.

This structured process ensures that the Wikipedia data is well-organized, easy to analyze, and available in both CSV and Markdown formats for different use cases.

### Running the Application
1. **Start the Streamlit Interface**:
   Launch the Q&A bot using:
   ```bash
   streamlit run app.py
   ```

2. **Interact with the Bot**:
   Enter historical queries and receive concise, contextually accurate answers.

### Example Query
**Input**: "What is the history of the Massachusetts Bay Colony?"  
**Output**: Founded in 1630, became Province of Massachusetts Bay in 1691–92

## Key Components

### Data Scraping
- Extracts content from Wikipedia.
- Converts processed content into CSV and Markdown formats for downstream use.

### Semantic Search and RAG Pipeline
- **Embeddings**: HuggingFace's `all-MiniLM-L6-v2` for semantic encoding.
- **Retriever**: FAISS for high-speed similarity search across document chunks.
- **Compressor**: LLMChainExtractor to condense retrieved content before passing it to the language model.

### Language Model
- **Model**: GPT-3.5-turbo via OpenAI API.
- **Function**: Generates natural language responses based on compressed, contextually relevant data.

## Dependencies
Key libraries and tools:
- `langchain`, `faiss-cpu`, `wikipedia-api`, `streamlit`, `openai`, `transformers`, `sentence-transformers`.

Refer to `requirements.txt` for the full list of dependencies.


## Troubleshooting
- **API Issues**: Ensure the OpenAI API key is valid and set up correctly in Streamlit secrets.
- **Vector Store**: Verify FAISS index is correctly initialized with the processed data.
- **UI Errors**: Restart the Streamlit server and check the console logs for detailed error messages.
