from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY não encontrada!")

numero_dias = 7
numero_criancas = 2
atividade = "praia"

modelo_de_prompt = PromptTemplate(
    template="""
    Crie um roteiro de viagem de {dias} dias,
    para uma família com {numero_criancas} crianças,
    que gostam de {atividade}
    """
)

prompt = modelo_de_prompt.format(
    dias=numero_dias,
    numero_criancas=numero_criancas,
    atividade=atividade
)

print("Prompt: \n", prompt)

prompt = f"Crie um roteiro simples (curto) de viagem de {numero_dias} dias, para uma família com {numero_criancas} crianças, que gosta de {atividade}."

cliente = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key
)

resposta = cliente.invoke([
    SystemMessage(
        content="Você é um assistente especializado em roteiros de viagens."
    ),
    HumanMessage(
        content=prompt
    )
])

if isinstance(resposta.content, str):
    resposta_em_texto = resposta.content
else:
    resposta_em_texto = "\n".join(
        bloco["text"]
        for bloco in resposta.content
        if isinstance(bloco, dict) and bloco.get("type") == "text"
    )

print(resposta_em_texto)