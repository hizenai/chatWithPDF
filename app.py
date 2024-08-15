import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
import glob

# from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    get_buffer_string,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from string import Template

# llm fr= OpenAI(temperature=0)
from langchain.chains import (
    StuffDocumentsChain,
    LLMChain,
    ConversationalRetrievalChain,
    ConversationChain,
    create_history_aware_retriever,
)

from langchain.chains.combine_documents import create_stuff_documents_chain
import json
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import random
import time

chat = ChatOpenAI(temperature=0)
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('intfloat/e5-large')

# Create the main function


def get_pdf_text(pdfs):
    raw_text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            raw_text += page.extract_text()
    return raw_text


def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# def get_conversational_rag_chain(chain):
#     rag_chain = create_stuff_documents_chain(chain)


def get_vector_store_from_url(url):
    loader = AsyncChromiumLoader([url])
    html = loader.load()
    return html


# Create the main function
def get_conversation_chain(vectorStore):

    llm = ChatOpenAI()

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
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()

    # chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | CONDENSE_QUESTION_PROMPT
    #     | llm
    #     | StrOutputParser()
    # )
    # memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorStore.as_retriever(),
    #     memory=memory,
    # )

    return conversational_qa_chain


def get_response(retrieval_text):

    return retrieval_text


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


def extract_entities_relationships(folder, prompt_template):
    files = glob.glob(f"./data/{folder}/*")
    system_message = "You are a helpful IT-project and account management expert who extracts information from documents."
    print(f"Running for pipeline for {len(files)} files in {folder} folder")
    results = []
    for i, file in enumerate(files):
        try:
            with open(file, "r") as f:
                text = f.read().rstrip()
                prompt = Template(prompt_template).substitute(ctext=text)
                result = process_gpt(prompt, system_message)
                results.append(json.loads(result))
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return results

def generate_cypher(json_object):
    entities = []
    relationships = []

    for i,obj in enumerate(json_object):
        for entity in obj['entities']:
            label = entity['label']
            Id = entity['id']
            properties = {k:v for k,v in entity.items() if k not in ['label','id']}

            cypher = f'MERGE (n: {label} {{id:"{Id}"}})'

            if properties:
                props_str = ','.join([f'n.{key} = "{val}"' for key,val in properties.items()])
                cypher += f"ON CREATE SET {props_str}"
            entities.append(cypher)
        for rel in obj['relationships']:
            start = rel['start']
            end = rel['end']


# Create the main function
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF", page_icon="📚")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello How Can I Help You?"),
            HumanMessage(content=""),
        ]
    # good practice

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.title("Chat with PDF")
    with st.sidebar:
        st.subheader("PDF file")
        st.write("This is the PDF file that you are chatting with.")
        pdfs = st.file_uploader(
            "Upload a PDF file", type="pdf", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdfs)
                text_chunks = get_chunks(raw_text)
                if "vectorStore" not in st.session_state:
                    st.session_state.vectorStore = get_vector_store(text_chunks)

        url = st.text_input("website url")

    if url is not None and url != "":
        st.info("Please enter a url")

    else:
        # documents = get_vector_store_from_url(url)
        # with st.sidebar:
        #     st.write(documents)
        # Accept user input
        if prompt := st.chat_input("What is up?"):
            conversation = get_conversation_chain(st.session_state.vectorStore)

            retrieved_texts = conversation.invoke(
                {
                    "question": prompt,
                    "chat_history": st.session_state.chat_history,
                }
            )

            response = get_response(retrieved_texts)

            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(response)

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                st.chat_message("AI")
                st.write(message.content)
            elif isinstance(message, HumanMessage):
                st.chat_message("Human")
                st.write(message.content)


if __name__ == "__main__":
    main()
