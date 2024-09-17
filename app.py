import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from operator import itemgetter

from PyPDF2 import PdfReader

# from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    get_buffer_string,
    SystemMessage,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# import recursive_character_text_splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# import prompt_template
from langchain_core.prompts import PromptTemplate

# llm fr= OpenAI(temperature=0)
from langchain.chains import (
    LLMChain,
)

from langchain_community.document_loaders import AsyncChromiumLoader

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
)
import os
import time

load_dotenv(dotenv_path="../chatWIthPdf/.env", override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key)
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('intfloat/e5-large')


# 1. Hierarchical Indexing
def create_hierarchical_index(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    return vector_store


# 2. Query Decomposition
def decompose_query(query):
    decomposition_template = """
    Break down the following complex query into simpler sub-queries:
    Query: {query}
    
    Output the sub-queries as a Python list.
    """
    decomposition_prompt = PromptTemplate(
        template=decomposition_template, input_variables=["query"]
    )
    decomposition_chain = LLMChain(llm=chat, prompt=decomposition_prompt)

    response = decomposition_chain.run(query=query)
    return eval(response)  # Convert string representation of list to actual list


# 3. Self-Reflection
def self_reflect(query, context, answer):
    reflection_template = """
    Analyze the following question, context, and answer:
    
    Question: {query}
    Context: {context}
    Answer: {answer}
    
    Reflect on the following:
    1. Is the answer directly addressing the question?
    2. Is the answer supported by the given context?
    3. Are there any logical inconsistencies in the answer?
    4. What additional information might be needed to provide a more comprehensive answer?
    
    Provide your reflection and any suggestions for improvement.
    """
    reflection_prompt = PromptTemplate(
        template=reflection_template, input_variables=["query", "context", "answer"]
    )
    reflection_chain = LLMChain(llm=chat, prompt=reflection_prompt)

    reflection = reflection_chain.run(query=query, context=context, answer=answer)
    return reflection


def get_pdf_text(pdfs):
    raw_text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            raw_text += page.extract_text()
    return raw_text


def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=800, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_vector_store_from_url(url):
    loader = AsyncChromiumLoader([url])
    html = loader.load()
    return html


# Create the main function
def get_conversation_chain(vectorStore):

    retriever = vectorStore.as_retriever()
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
            Chat History: {chat_history}
            Follow Up Input: {question}
            Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    _inputs = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    )
    _context = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }
    conversational_qa_chain = (
        _inputs
        | _context
        | ANSWER_PROMPT
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
    )

    return conversational_qa_chain


def get_response(retrieval_text):

    response_template = """
    Based on the following retrieved information, provide a comprehensive and accurate response:

    {retrieval_text}

    Response:
    """
    response_prompt = PromptTemplate(
        template=response_template, input_variables=["retrieval_text"]
    )
    response_chain = LLMChain(llm=chat, prompt=response_prompt)

    response = response_chain.run(retrieval_text=retrieval_text)
    return response


def typing_effect(message):
    for char in message:
        st.write(char, end="", flush=True)
        time.sleep(0.05)  # Adjust the sleep time as needed for desired typing speed
    st.write("")  # To move to the next line after typing completes


# Streamed response emulator
def response_generator(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.write(response)


def process_gpt(content, prompt):

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=content),
    ]

    return chat.invoke(messages)


# summarize
def summarize_text(text):
    prompt = f"""
    Provide a comprehensive and detailed summary of the following text, with a focus on key points, methodologies, and findings. If the text contains technical details, ensure they are accurately represented in the summary. For research papers, highlight the research question, methodology, results, and conclusions. For financial reports, emphasize important financial metrics, trends, and any significant changes or projections.

    Text to summarize:
    {text}

    Detailed Summary:
    """

    response = chat.predict(prompt)
    summary = response.strip()
    return summary


def summarize_long_text(text, max_chunk_tokens=8000):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""
        Provide a detailed summary of the following text chunk, which is part {i+1} of a larger document. Focus on key points, methodologies, findings, and any technical details. Ensure that important information, especially numerical data or specific technical terms, are accurately captured.

        Text chunk:
        {chunk}

        Detailed Summary of Chunk {i+1}:
        """

        response = chat.predict(prompt)
        summaries.append(response.strip())
    # Combine the summaries
    combined_summary = "\n\n".join(summaries)
    # Optionally, summarize the combined summary if it's still too long
    if len(combined_summary) > max_chunk_tokens:
        prompt = f"""
        Synthesize a final, comprehensive summary from the following chunk summaries of a larger document. Ensure that all key points, methodologies, findings, and technical details are accurately represented. Pay special attention to maintaining the integrity of any numerical data, specific technical terms, or critical insights from the original text.

        Chunk Summaries:
        {combined_summary}

        Final Comprehensive Summary:
        """
        final_summary = chat.predict(prompt).strip()
        return final_summary
    else:
        return combined_summary


def get_enhanced_conversation_chain(vectorStore):
    retriever = vectorStore.as_retriever()

    def enhanced_qa(query):
        # Query decomposition
        sub_queries = decompose_query(query)

        # Retrieve information for each sub-query
        sub_answers = []
        for sub_query in sub_queries:
            docs = retriever.get_relevant_documents(sub_query)
            context = "\n".join([doc.page_content for doc in docs])
            sub_answer = get_response(context)
            sub_answers.append(sub_answer)

        # Combine sub-answers
        combined_answer = "\n\n".join(sub_answers)

        # Generate final answer
        final_answer = get_response(combined_answer)

        # Self-reflection
        reflection = self_reflect(query, combined_answer, final_answer)

        return {"answer": final_answer, "reflection": reflection}

    return enhanced_qa


# Streamlit App


def main():

    st.set_page_config(page_title="Self-Reflective RAG Chatbot", page_icon="🤖")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! How can I help you today?"),
            HumanMessage(content=""),
        ]

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    st.title("Self-Reflective RAG Chatbot")

    with st.sidebar:
        st.subheader("Upload Documents")
        pdfs = st.file_uploader(
            "Upload PDF files", type="pdf", accept_multiple_files=True
        )
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(pdfs)
                st.session_state.retriever = create_hierarchical_index(raw_text)
                st.session_state.raw_text = raw_text
                st.success("Documents processed successfully!")
        if st.button("Generate Summary"):
            # parse the text
            raw_text = get_pdf_text(pdfs)
            with st.spinner("Generating summary..."):
                summary = summarize_long_text(raw_text)
                st.session_state.summary = summary
                st.success("Summary generated successfully!")

    if "summary" in st.session_state:
        st.subheader("Document Summary")
        st.write(st.session_state.summary)

    if st.session_state.retriever is None:
        st.info("Please upload and process documents to start chatting.")
    else:
        if prompt := st.chat_input("Ask a question about the uploaded documents"):
            st.session_state.chat_history.append(HumanMessage(content=prompt))

            conversation = get_enhanced_conversation_chain(st.session_state.retriever)

            with st.spinner("Thinking..."):
                response = conversation(prompt)

            st.session_state.chat_history.append(AIMessage(content=response["answer"]))
            st.subheader("AI Response:")
            st.write(response["answer"])

            with st.expander("View AI's Self-Reflection"):
                st.write(response["reflection"])

        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                st.chat_message("AI").write(message.content)
            elif isinstance(message, HumanMessage):
                st.chat_message("Human").write(message.content)


if __name__ == "__main__":
    main()
