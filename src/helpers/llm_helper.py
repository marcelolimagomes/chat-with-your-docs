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

# Define os diretórios utilizados pelo aplicativo
data_dir = './data'
documents_dir = f'{data_dir}/documents'
indexed_files_dir = f'{documents_dir}/indexed_files'
database_dir = f'{data_dir}/database'

# Recupera o diretório do banco de dados vectorstore


def get_database_dir(session_id):
    return f"{database_dir}/{session_id}"

# Recupera o arquivo do banco de dados vectorstore


def get_database_file(session_id):
    return f"{get_database_dir(session_id)}/index.faiss"

# Converte uma imagem em base64 para que seja exibida na página


@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Define a imagem de fundo da página


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

# Valida se o modelo selecionado pelo usuário está disponível no servidor Ollama configurado


def is_model_available(model_name):
    models = ollama.list()
    return any(model_name in model['model'] for model in models['models'])

# Unifica os documentos recuperados do bando de dados vectorstore


def get_unique_union(documents: list[list]):
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Inicia o chatbot com o modelo de linguagem selecionado pelo usuário
# Neste caso o modelo implementa um RAG
# Com base na técnica de Query Transformation utilizando LangChain


def chat_rag(user_prompt, model, database_dir):
    print(f'DATABASE_DIR:', database_dir)
    vectorize = get_faiss(database_dir, get_embedding())  # Carrega o vetorstore salvo em disco
    print('VECTORSTORE:', vectorize)

    prompt_perspectives = ChatPromptTemplate.from_template(Config.SYSTEM_PROMPT_RAG)  # Cria um prompt com base no template

    llm = OllamaLLM(model=model)  # Inicializa o modelo de linguagem utilizando Ollama

    # Cadeia de processamento para gerar novas perguntas
    generate_queries = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    # Cadeia de processamento para recuperar documentos relevantes
    retrieval_chain = generate_queries | vectorize.as_retriever().map() | get_unique_union

    # Customiza o prompt para que o modelo de LLM foque no contexto fornecido para gerar respostas mais precisas
    # Esse é a base do conceito de RAG (Retrieval Augmented Generation)
    template = """Responda a seguinte pergunta com base no contexto fornecido:

    {context}

    Pergunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)  # Cria um prompt com base no template

    # Cadeia de processamento final para gerar a resposta do modelo
    final_rag_chain = (
        {"context": retrieval_chain,
         "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )

    return final_rag_chain.invoke({"question": user_prompt})  # Invoca a cadeia de processamento para gerar a resposta


# Inicia o chatbot com o modelo de linguagem selecionado pelo usuário
# Neste caso o modelo implementa um LLM puro, não implementa RAG
def chat_llm_pure(user_prompt, model):
    stream = ollama.chat(
        model=model,
        messages=[{'role': 'assistant', 'content': Config.SYSTEM_PROMPT_LLM},
                  {'role': 'user', 'content': user_prompt}],
        stream=True,
    )

    return stream


def stream_parser(stream):  # Função para processar o stream de mensagens retornados pelo Ollama
    for chunk in stream:
        yield chunk['message']['content']

# Trata o texto gerado pelo modelo


def clean_text(text):
    while '  ' in text:
        text = text.replace('  ', ' ')
    for s in string.punctuation:
        text = text.replace(s + s, s)
    return text.strip()

# Extrai o texto de um arquivo PDF


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

# Processa um lote de arquivos PDF e armazena o texto extraído em arquivos TXT
# Os arquivos são gravados no diretório 'indexed' dentro do diretório de origem


def process_batch_pdf(path_to_pdf_files):
    list_pdf_files = glob.glob(path_to_pdf_files + '/*.pdf')
    for pdf_file in list_pdf_files:
        print(f'Extracting text from {pdf_file} ...')
        lines = extract_text_from_pdf(pdf_file)
        with open(pdf_file.replace('pdf', 'txt'), 'w') as f:
            f.writelines(lines)
        if not os.path.exists(f'{path_to_pdf_files}/indexed'):
            os.makedirs(f'{path_to_pdf_files}/indexed')
        filename = get_filename(pdf_file)
        os.rename(pdf_file, f'{path_to_pdf_files}/indexed/{filename}')

# Recupera o diretório de upload de arquivos


def get_upload_dir(session_id):
    return f"{documents_dir}/upload_content_" + session_id

# Limpa os arquivos enviados pelo usuário


def get_uploaded_files(session_id):
    upload_dir = f'{get_upload_dir(session_id)}/indexed'
    files = [file.split('/')[-1] for file in glob.glob(f"{upload_dir}/*.pdf")]
    return pd.DataFrame(data=files, columns=['Arquivos enviados:'])

# Realiza o upload de um arquivo enviado pelo usuário


def upload_file(session_id, uploaded_file):
    # Save uploaded file to 'upload_content' directory
    upload_dir = get_upload_dir(session_id)
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return upload_dir

# Recupera o banco de dados vectorstore


def get_faiss(database_dir, embeddings):
    if os.path.isfile(f'{database_dir}/index.faiss'):
        vectorstore = FAISS.load_local(database_dir, embeddings, allow_dangerous_deserialization=True)  # Carrega o vetorstore salvo em disco
        return vectorstore
    return None

# Recupera o modelo de embeddings e armazena na sessão


@st.cache_resource(ttl=600)
def get_embedding():
    embeddings = OllamaEmbeddings(model="mistral")  # Usa o modelo "mistral" para embeddings
    return embeddings

# Indexa os documentos enviados pelo usuário no banco de dados vectorstore


def vectorize(path_to_txt_files, path_database):
    new_documents = []
    list_txt_files = glob.glob(f'{path_to_txt_files}/*.txt')
    for txt_file in list_txt_files:
        loader = TextLoader(txt_file)
        new_documents.extend(loader.load())
        indexed_files_dir
        if not os.path.exists(f'{path_to_txt_files}/indexed'):
            os.makedirs(f'{path_to_txt_files}/indexed')
        filename = get_filename(txt_file)
        os.rename(txt_file, f'{path_to_txt_files}/indexed/{filename}')

    if len(new_documents) == 0:
        return None

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(new_documents)
    new_vectorstore = FAISS.from_documents(texts, get_embedding())

    vectorstore = get_faiss(path_database, get_embedding())
    if vectorstore is not None:
        vectorstore.merge_from(new_vectorstore)
        vectorstore.save_local(path_database)
    else:
        vectorstore = new_vectorstore
        vectorstore.save_local(path_database)
    return vectorstore

# Limpa os arquivos enviados pelo usuário


def get_filename(filepath):
    return os.path.basename(filepath)

# Recupera o diretório de upload de arquivos


def get_dirname(filepath):
    return os.path.dirname(os.path.abspath(filepath))
