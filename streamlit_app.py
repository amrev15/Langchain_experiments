import streamlit as st
import pdfplumber  # For reading PDF files
import pandas as pd  # For reading CSV files
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def extract_text_from_file(uploaded_file):
    """Extracts text from PDF, TXT, or CSV files."""

    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages)
    elif file_extension in ("txt", "csv"):
        text = uploaded_file.read().decode()
        if file_extension == "csv":
            # Convert CSV to plain text
            df = pd.read_csv(uploaded_file)
            text = df.to_string(index=False)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    return text

def generate_response(uploaded_file, openai_api_key, query_text):
    """Generates a response to the query using the uploaded document."""

    if uploaded_file is not None:
        text = extract_text_from_file(uploaded_file)
        documents = [text]

        # Remaining code remains the same
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=openai_api_key), chain_type="stuff", retriever=retriever
        )
        return qa.run(query_text)

# Page title
st.set_page_config(page_title=' Ask the Doc App')
st.title(' Ask the Doc App')

# File upload - allowing PDF, TXT, and CSV
uploaded_file = st.file_uploader('Upload an article', type=['pdf', 'txt', 'csv'])
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
