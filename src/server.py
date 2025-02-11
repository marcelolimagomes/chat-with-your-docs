import streamlit as st
import os

from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
from config import Config
from helpers.llm_helper import *

st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon="./img/logo.jpg",
    initial_sidebar_state="expanded",
)

ctx = get_script_run_ctx()


@st.cache_data(ttl=600)
def get_session_id():
    # Initialization
    if 'id_do_usuario' not in st.session_state:
        st.session_state['id_do_usuario'] = ctx.session_id
        result = st.session_state['id_do_usuario']
    else:
        result = st.session_state['id_do_usuario']
    return result


session_id = get_session_id()
print(f"Session ID: {session_id}")

set_png_as_page_bg('./img/logo.png')
st.image('./img/logo.jpg')

# sets up sidebar nav widgets
with st.sidebar:
    st.markdown('# ' + Config.PAGE_TITLE)

    # widget - https://docs.streamlit.io/library/api-reference/widgets/st.selectbox
    model = st.selectbox('Qual o modelo você gostaria de usar?', Config.OLLAMA_MODELS)
    df_arquivos = get_uploaded_files(session_id)
    df_arquivos.index.name = "#"
    st.table(df_arquivos)

# checks for existing messages in session state
# https://docs.streamlit.io/library/api-reference/session-state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from session state
# https://docs.streamlit.io/library/api-reference/chat/st.chat_message
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input for user prompt
user_prompt = st.chat_input("Me faça uma pergunta ou envie seu documento PDF!")

if user_prompt:
    # Display user prompt in chat message widget
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # adds user's prompt to session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.spinner('Gerando a resposta...'):
        # retrieves response from model
        print('PERGUNTA:> ', user_prompt)
        llm_stream = chat(user_prompt, model=model, database_dir=get_database_dir(session_id))

        # streams the response back to the screen
        # stream_output = st.write_stream(stream_parser(llm_stream))
        stream_output = st.write(llm_stream)
        print('RESPOSTA:> ', stream_output)

        # appends response to the message list
        st.session_state.messages.append({"role": "assistant", "content": stream_output})

# File uploader for TXT and PDF files
uploaded_file = st.file_uploader("Envie seu documento PDF", type=["pdf"])
if uploaded_file is not None:
    upload_dir = upload_file(session_id, uploaded_file)
    st.success(f"Arquivo {uploaded_file.name} enviado com sucesso. Indexação em andamento...")
    process_batch_pdf(upload_dir)
    vectorize(upload_dir, get_database_dir(session_id))
    st.success(f"Indexação concluída! Faça uma pergunta relacionada ao documento.")
