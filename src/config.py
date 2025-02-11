"""
Orignal Author: DevTechBytes
https://www.youtube.com/@DevTechBytes
"""


class Config:
    PAGE_TITLE = "Multi-Model Chat"

    OLLAMA_MODELS = ('llama3.1:8b', 'llama3.2:1b', 'llama3.2:3b',
                     'deepseek-r1:1.5b', 'deepseek-r1:7b', 'deepseek-r1:8b', 'deepseek-r1:14b',
                     'phi4', 'mistral:7b', 'olmo2:7b',
                     'codellama:7b', 'codellama:13b', 'llama2-uncensored',
                     )

    SYSTEM_PROMPT = """Você é um assistente de modelo de linguagem de IA. Sua tarefa é gerar cinco
versões diferentes da pergunta do usuário fornecida para recuperar documentos relevantes de um vector database. 
Ao gerar múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar
o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
Forneça essas perguntas alternativas separadas por quebras de linha. Pergunta original: {question}"""
