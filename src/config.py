"""
Configurações do aplicativo.
"""


class Config:
    PAGE_TITLE = "Multi-Model RAG Chatbot"  # Título da página

    # Modelos de linguagem disponíveis
    # O download dos modelos de linguagem é feito em tempo de execução quando não estão disponíveis localmente
    OLLAMA_MODELS = ('llama3.1:8b', 'llama3.2:1b', 'llama3.2:3b',
                     'deepseek-r1:1.5b', 'deepseek-r1:7b', 'deepseek-r1:8b', 'deepseek-r1:14b',
                     'phi4', 'mistral:7b', 'olmo2:7b',
                     'codellama:7b', 'codellama:13b', 'llama2-uncensored',
                     )

    # Tecnicas de geração de novas perguntas baseado na pergunta original
    # Query transformation são um conjunto de abordagens focadas em reescrever e/ou modificar
    # perguntas para recuperação.
    # Este prompt é utilizado somente quando o usuário envia documentos para o chatbot
    SYSTEM_PROMPT_RAG = """Você é um assistente de modelo de linguagem de IA. Sua tarefa é gerar cinco
versões diferentes da pergunta do usuário fornecida para recuperar documentos relevantes de um vector database. 
Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar
o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
Forneça essas perguntas alternativas separadas por quebras de linha. Pergunta original: {question}"""

    # Criação de contexto para que o modelo de linguagem possa gerar respostas mais precisas
    # Esse prompt é utilizado quando o usuáiro **não** envia documentos para o chatbot
    SYSTEM_PROMPT_LLM = """Você é um chat bot baseado em Inteligência Artifical Generativa. 
Sua tarefa é responder a pergunta do usuário de forma detalhada e com múltiplas perspectivas.
Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar
o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
Forneça respostas alternativas separadas por quebras de linha com base na pergunta."""
