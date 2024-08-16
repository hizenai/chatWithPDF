# Self-Reflective RAG Chatbot

A Streamlit-based chatbot that uses advanced Retrieval-Augmented Generation (RAG) techniques, including self-reflection, query decomposition, and hierarchical indexing, to provide intelligent responses based on uploaded PDF documents.

## Features

- **PDF Document Processing**: Extract and process text from multiple PDF files.
- **Hierarchical Indexing**: Utilize a two-level indexing system for more accurate information retrieval.
- **Query Decomposition**: Break down complex queries into simpler sub-queries for comprehensive answers.
- **Self-Reflection**: Analyze and reflect on the generated answers for improved accuracy and transparency.
- **Streamlit User Interface**: Easy-to-use web interface for document uploading and chatting.

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/self-reflective-rag-chatbot.git
   cd self-reflective-rag-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the sidebar to upload PDF documents and process them.

4. Once the documents are processed, you can start chatting with the bot about the content of your documents in the main chat interface.

## How it Works

1. **Document Processing**: 
   - The app extracts text from uploaded PDF files.
   - The text is split into parent and child chunks using `RecursiveCharacterTextSplitter`.

2. **Hierarchical Indexing**:
   - Child chunks are embedded and stored in a FAISS vector store.
   - Parent chunks are stored in an in-memory document store.
   - A `ChildParentRetriever` is used to fetch relevant documents based on queries.

3. **Query Processing**:
   - Complex queries are broken down into simpler sub-queries using an LLM.
   - Each sub-query is processed separately, retrieving relevant context from the hierarchical index.

4. **Answer Generation**:
   - The LLM generates answers for each sub-query based on the retrieved context.
   - Sub-answers are combined to form a comprehensive response.

5. **Self-Reflection**:
   - The chatbot analyzes its own answer, considering factors like relevance, consistency, and completeness.
   - A reflection is generated, providing insights into the answer's quality and potential areas for improvement.

6. **User Interaction**:
   - The Streamlit interface displays the chatbot's answer and allows users to view its self-reflection.
   - The chat history is maintained for context in future queries.

## Customization

You can customize various aspects of the chatbot by modifying the following in `app.py`:
- Chunk sizes and overlap in the `create_hierarchical_index` function
- Prompt templates for query decomposition, answer generation, and self-reflection
- LLM parameters (e.g., temperature) in the `ChatOpenAI` instances

## Limitations

- The chatbot's knowledge is limited to the content of the uploaded PDFs.
- Processing large or numerous PDFs may take considerable time and computational resources.
- The quality of responses depends on the OpenAI API and the specificity of the user's questions.

## Contributing

Contributions to improve the chatbot are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.
