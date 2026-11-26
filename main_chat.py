import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key
)

prompt_sugestao = ChatPromptTemplate.format_messages(
    [
        ("system", "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Destinos")
    ]
)

lista_pergunta = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

for uma_pergunta in lista_pergunta:
    resposta = modelo.invoke(uma_pergunta)
    print("Usuário: ", uma_pergunta)
    print("IA: ", resposta.content, "\n")