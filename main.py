from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

class Destino(BaseModel):
    cidade:str = Field(description="A cidade recomendada para visitar")
    motivo:str = Field(description="O motivo pelo qual é interessante visitar essa cidade")

class Restaurantes(BaseModel):
    cidade:str = Field(description="A cidade recomendada para visitar")
    restaurantes:str = Field(description="Restaurantes recomendados na cidade")

parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

if not api_key:
    raise ValueError("GOOGLE_API_KEY não encontrada!")

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.

    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()}
)

prompt_restaurantes = PromptTemplate(
    template="""
    Sugira restaurantes populares entre locais em {cidade}
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()}
)

prompt_cultural = PromptTemplate(
    template="Sugira atividades e locais culturais em {cidade}"
)

modelo = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key
)

cadeia_1 = prompt_cidade | modelo | parseador_destino
cadeia_2 = prompt_restaurantes | modelo | parseador_restaurantes
cadeia_3 = prompt_cultural | modelo | StrOutputParser()

cadeia = (cadeia_1 | cadeia_2 | cadeia_3)

resposta = cadeia.invoke({
    "interesse": "praia"
})

print(resposta)