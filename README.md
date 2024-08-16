# Self-Reflective RAG Chatbot

An advanced chatbot leveraging LangChain, GPT, and sophisticated retrieval techniques for dynamic interactions with PDF and web content. This project combines a Streamlit-based web application with a knowledge graph backend to create a powerful, context-aware chatbot.

## Features

- **Self-Reflective RAG (Retrieval-Augmented Generation)**: Utilizes LangChain and GPT for intelligent, context-aware responses.
- **PDF and Web Content Processing**: Extracts and processes information from uploaded PDFs and specified web URLs.
- **Dynamic Conversation Chain**: Implements a conversational retrieval chain for maintaining context across interactions.
- **Vector Store Integration**: Uses FAISS for efficient similarity search and retrieval.
- **Streamlit Web Interface**: Provides an intuitive user interface for interacting with the chatbot.
- **Knowledge Graph Backend**: Utilizes Neo4j for storing and querying complex relationships extracted from documents.

## Project Structure

- `app.py`: Main Streamlit application file containing the chatbot interface and core logic.
- `knowledge-graph.ipynb`: Jupyter notebook for knowledge graph operations and GPT processing.

## Installation

1. Clone this repository:
   ```
   git clone [repository-url]
   cd [repository-name]
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   OPENAI_API_KEY=your_openai_api_key
   NEO4J_CONNECTION_URI=your_neo4j_uri
   NEO4J_USER=your_neo4j_username
   NEO4J_PASSWORD=your_neo4j_password
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Upload PDF files or enter a website URL in the sidebar.

3. Interact with the chatbot by typing your questions in the chat input.

## Dependencies

- streamlit
- python-dotenv
- PyPDF2
- langchain
- openai
- faiss-cpu
- neo4j
- beautifulsoup4
- chromadb

## Configuration

The project uses environment variables for configuration. Ensure you have set up the `.env` file with the necessary API keys and database credentials as mentioned in the Installation section.

## Development

To work on the knowledge graph component:

1. Open `knowledge-graph.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure you have the required environment variables set.
3. Run the cells to process documents and interact with the Neo4j database.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
