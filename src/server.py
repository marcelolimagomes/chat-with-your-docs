from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 1️⃣ Carregar documentos (pode substituir pelo seu próprio dataset)
loader = TextLoader("data/CF_EC1342024.txt")  # Carregue um arquivo local
documents = loader.load()

# 2️⃣ Dividir o texto em pedaços menores para indexação eficiente
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3️⃣ Criar embeddings com Ollama
embeddings = OllamaEmbeddings(model="mistral")  # Usa o modelo "mistral" para embeddings

# 4️⃣ Criar e armazenar a base de dados vetorial com FAISS
vectorstore = FAISS.from_documents(texts, embeddings)

# 5️⃣ Carregar o modelo Llama 3.1 com Ollama para geração de texto
llm = Ollama(model='deepseek-r1:1.5b')  # "llama3")

# 6️⃣ Criar o sistema de Perguntas e Respostas (RAG)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

# 7️⃣ Fazer uma consulta ao RAG
query = "Quais são as regras para que um trabalhador no regime CLT tenha direito a férias?"
response = qa_chain.run(query)

# 8️⃣ Exibir resposta gerada
print(response)
