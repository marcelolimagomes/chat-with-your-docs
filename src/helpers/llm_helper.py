"""
Orignal Author: DevTechBytes
https://www.youtube.com/@DevTechBytes
"""

# importing required modules
from pypdf import PdfReader
from config import Config
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from operator import itemgetter

import ollama
import streamlit as st
import base64
import string
import os
import glob
import pandas as pd

system_prompt = Config.SYSTEM_PROMPT
data_dir = './data'
documents_dir = f'{data_dir}/documents'
indexed_files_dir = f'{documents_dir}/indexed_files'
database_dir = f'{data_dir}/database'


def get_database_dir(session_id):
    return f"{database_dir}/{session_id}"


def get_database_file(session_id):
    return f"{get_database_dir(session_id)}/index.faiss"


@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center center;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


def is_model_available(model_name):
    models = ollama.list()
    return any(model_name in model['model'] for model in models['models'])


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def chat(user_prompt, model, database_dir):
    print(f'DATABASE_DIR:', database_dir)
    vectorize = get_faiss(database_dir, get_embedding())
    print('VECTORSTORE:', vectorize)

    prompt_perspectives = ChatPromptTemplate.from_template(Config.SYSTEM_PROMPT)

    llm = OllamaLLM(model=model)
    generate_queries = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    retrieval_chain = generate_queries | vectorize.as_retriever().map() | get_unique_union
    # docs = retrieval_chain.invoke({"question": user_prompt})

    # RAG
    template = """Responda a seguinte pergunta com base no contexto fornecido:

    {context}

    Pergunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        {"context": retrieval_chain,
         "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )

    return final_rag_chain.invoke({"question": user_prompt})

    stream = ollama.chat(
        model=model,
        messages=[{'role': 'assistant', 'content': system_prompt},
                  {'role': 'user', 'content': f"Model being used is {model}.{user_prompt}"}],
        stream=True,
    )

    return stream


def stream_parser(stream):
    for chunk in stream:
        yield chunk['message']['content']


def clean_text(text):
    while '  ' in text:
        text = text.replace('  ', ' ')
    for s in string.punctuation:
        text = text.replace(s + s, s)
    return text.strip()


def extract_text_from_pdf(file_path):
    if '.pdf' not in file_path:
        raise Exception(f'File {file_path} is not a pdf file!')
    result = []
    # creating a pdf reader object
    reader = PdfReader(file_path)
    # getting a specific page from the pdf file
    for page in reader.pages:
        # extracting text from page
        text = page.extract_text()
        text = clean_text(text)
        if len(text.strip()) == 0:
            continue
        result.append(text)
    return ' '.join(result)


def process_batch_pdf(path_to_pdf_files):
    list_pdf_files = glob.glob(path_to_pdf_files + '/*.pdf')
    for pdf_file in list_pdf_files:
        print(f'Extracting text from {pdf_file} ...')
        lines = extract_text_from_pdf(pdf_file)
        with open(pdf_file.replace('pdf', 'txt'), 'w') as f:
            f.writelines(lines)


def get_updated_dir(session_id):
    return f"{documents_dir}/upload_content_" + session_id


def get_uploaded_files(session_id):
    upload_dir = get_updated_dir(session_id)
    files = [file.split('/')[-1] for file in glob.glob(f"{upload_dir}/*.pdf")]
    return pd.DataFrame(data=files, columns=['Arquivos enviados:'])


def upload_file(session_id, uploaded_file):
    # Save uploaded file to 'upload_content' directory
    upload_dir = get_updated_dir(session_id)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return upload_dir


def get_faiss(database_dir, embeddings):
    if os.path.isfile(f'{database_dir}/index.faiss'):
        vectorstore = FAISS.load_local(database_dir, embeddings, allow_dangerous_deserialization=True)  # Carrega o vetorstore salvo em disco
        return vectorstore
    return None


@st.cache_resource(ttl=600)
def get_embedding():
    # 3️⃣ Criar embeddings com Ollama
    embeddings = OllamaEmbeddings(model="mistral")  # Usa o modelo "mistral" para embeddings
    return embeddings


def vectorize(txt_data_path, path_database):
    # 🔹 Load and process new documents
    new_documents = []
    list_txt_files = glob.glob(f'{txt_data_path}/*.txt')
    for txt_file in list_txt_files:
        loader = TextLoader(txt_file)
        new_documents.extend(loader.load())
        indexed_files_dir
        os.rename(txt_file, txt_file.replace('txt', 'indexed'))

    if len(new_documents) == 0:
        return None

    # 2️⃣ Dividir o texto em pedaços menores para indexação eficiente
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(new_documents)
    # 🔹 Generate embeddings
    new_vectorstore = FAISS.from_documents(texts, get_embedding())

    vectorstore = get_faiss(path_database, get_embedding())
    if vectorstore is not None:
        vectorstore.merge_from(new_vectorstore)
        vectorstore.save_local(path_database)
    else:
        vectorstore = new_vectorstore
        vectorstore.save_local(path_database)
    return vectorstore
