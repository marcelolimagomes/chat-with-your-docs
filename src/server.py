import streamlit as st
import os
import time

from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
from config import Config
from helpers.llm_helper import *

# -- Configurações da página

# Configura a página do Streamlit com título, ícone e estado inicial da barra lateral
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon="./img/logo.jpg",
    initial_sidebar_state="expanded",
)

# Define a imagem de fundo da página
set_png_as_page_bg('./img/logo.jpg')
# Exibe a imagem do logo na página
st.image('./img/logo.jpg')
# Obtém o contexto da execução do script
ctx = get_script_run_ctx()

# -- Funções auxiliares

# Função para obter o ID da sessão do usuário, com cache de 600 segundos


@st.cache_data(ttl=600)
def get_session_id():
    # Inicializa o ID do usuário na sessão se não existir
    if 'id_do_usuario' not in st.session_state:
        st.session_state['id_do_usuario'] = ctx.session_id
        result = st.session_state['id_do_usuario']
    else:
        result = st.session_state['id_do_usuario']
    return result


# Obtém o ID da sessão do usuário
session_id = get_session_id()
print(f"Session ID: {session_id}")

# Função para exibir uma caixa de diálogo de validação


@st.dialog("Mensagem do Sistema!")
def validate_box(message, callback_yes, callback_no, **kwargs):
    st.write(message)
    if st.button("Sim"):
        callback_yes(kwargs)
    if st.button("Não"):
        if callback_no:
            callback_no(kwargs)

# Função para remover arquivos associados à sessão do usuário


def remove_files(kwargs):
    session_id = kwargs.get("session_id")
    clear_chat()
    try:
        print(os.system(f"rm -rf {get_database_dir(session_id)}"))
        print(os.system(f"rm -rf {get_upload_dir(session_id)}"))
        st.success("Arquivos removidos com sucesso!")
    except Exception as e:
        st.error(f"Erro ao remover arquivos: {e}")
    time.sleep(1)
    st.rerun()

# Função para limpar a tela


def clear(id):
    st.rerun()

# Função para limpar arquivos da sessão do usuário


def clear_files(session_id):
    validate_box("Deseja realmente remover todos os arquivos?", remove_files, clear, session_id=session_id)
    st.rerun()

# Função para fazer upload de arquivos PDF


def upload():
    uploaded_file = st.file_uploader("Envie seu documento PDF", key=st.session_state['upload_key'], type=["pdf"])
    if uploaded_file is not None:
        clear_chat()
        print('>>> Uploaded file:', uploaded_file)
        upload_dir = upload_file(session_id, uploaded_file)
        st.warning(f"Arquivo {uploaded_file.name} enviado com sucesso. Indexação em andamento. Isso pode demorar...")
        process_batch_pdf(upload_dir)
        vectorize(upload_dir, get_database_dir(session_id))
        st.success(f"Indexação concluída! Faça uma pergunta relacionada ao documento.")
        st.session_state['upload_key'] = f"{time.time_ns()}"
        time.sleep(2)
        st.rerun()

# Função para limpar o chat


def clear_chat():
    st.session_state.messages = []

# Função principal que trata o comportamento da página


def __main__():

    # -- Início do código que trata o comportamento da página
    if "messages" not in st.session_state:
        clear_chat()

    # Exibe as mensagens do chat armazenadas na sessão
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["content"] is not None:
                st.markdown(message["content"])

    # Inicializa a chave de upload se não existir
    if 'upload_key' not in st.session_state:
        st.session_state['upload_key'] = f"{time.time_ns()}"

    # Configura a barra lateral
    with st.sidebar:
        st.markdown('# ' + Config.PAGE_TITLE)

        # Exibe a tabela de arquivos enviados
        model = st.selectbox('Qual o modelo você gostaria de usar?', Config.OLLAMA_MODELS, on_change=clear_chat)
        if not is_model_available(model):  # Se o modelo não estiver disponível em cache, baixa o modelo
            try:
                st.warning(f"Modelo {model} não disponível em cache. Estamos baixando o modelo. Isso pode demorar...")
                ollama.pull(model)  # Baixa o modelo
                st.success(f"Modelo {model} baixado com sucesso!")
                clear_chat()
            except Exception as e:  # Exibe mensagem de erro se houver falha no download
                st.error(f"Erro ao baixar modelo. Tente outro ou volte mais tarde. Msg: {e}")
        df_arquivos = get_uploaded_files(session_id)  # Obtém os arquivos enviados
        df_arquivos.index.name = "#"  # Define o nome do índice da tabela
        st.table(df_arquivos)  # Exibe a tabela de arquivos enviados
        if df_arquivos.shape[0] > 0:  # Se existirem arquivos enviados, exibe o botão para limpar arquivos
            st.button('Limpar arquivos', on_click=clear_files, args=(session_id,))

        st.markdown('---')
        upload()  # Exibe o botão para upload de arquivos
        st.markdown('---')
        st.markdown("""
### Desenolvido por: *Marcelo Lima Gomes*
### [Linkedin](https://www.linkedin.com/in/marcelolimagomes/)
### [GitHub](https://github.com/marcelolimagomes)
### [Email](mailto:marcelolimagomes@gmail.com)
""")

    print('SESSION:\n', st.session_state)

    # Entrada de chat para o usuário
    user_prompt = st.chat_input("Me faça uma pergunta ou envie seu documento PDF!")

    if user_prompt:  # Se o usuário enviar uma pergunta
        # Exibe a pergunta do usuário no chat
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Caso existam arquivos enviados, utiliza o RAG para responder
        if df_arquivos.shape[0] > 0:
            with st.spinner('Gerando a resposta utilizando utilizando o RAG dos documentos enviados...'):
                # Obtém a resposta do modelo
                print('Pergunta: ', user_prompt)
                llm_stream = chat_rag(user_prompt, model=model, database_dir=get_database_dir(session_id))
                with st.chat_message("assistant"):
                    stream_output = st.write(llm_stream)
        else:  # Caso contrário, utiliza o LLM puro
            if len(st.session_state.messages) > 0:  # Se existirem perguntas anteriores, exibe-as
                # Monta o prompt com as perguntas anteriores e a nova pergunta
                # Essa tecnica é conhecida como "Prompt Engineering"
                # Permite que o modelo tenha mais contexto para gerar uma resposta mais precisa
                # baseado nas perguntas anteriores
                final_prompt = 'Utilize a lista de PERGUNTAS ANTERIORES para responder a NOVA PERGUNTA.\n'
                count = 1
                for message in st.session_state.messages:  # Exibe as perguntas anteriores
                    if message["role"] == "user":
                        final_prompt += f'PERGUNTA ANTERIOR {count}: {message["content"]}\n'
                        count += 1
                final_prompt += 'NOVA PERGUNTA: ' + user_prompt
            else:  # Caso contrário, exibe apenas a nova pergunta
                final_prompt = user_prompt
            with st.spinner('Gerando a resposta utilizando o modelo LLM puro, sem RAG. Selecione um arquivo para ativar o RAG.'):
                # Obtém a resposta do modelo
                print('Pergunta: ', final_prompt)
                llm_stream = chat_llm_pure(final_prompt, model=model)
                with st.chat_message("assistant"):
                    stream_output = st.write_stream(stream_parser(llm_stream))

        # adiciona a pergunta do usuário à lista de mensagens
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # adiciona a resposta do modelo à lista de mensagens
        st.session_state.messages.append({"role": "assistant", "content": stream_output})

    # Botão para limpar o chat
    if len(st.session_state.messages) > 0:
        st.button('Limpar chat', on_click=clear_chat, args=())


# -- Execução do código principal
__main__()
