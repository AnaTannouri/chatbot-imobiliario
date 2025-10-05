from flask import Flask, request
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from telegram import Bot
from dotenv import load_dotenv
import os, re

# === CARREGAR VARIÁVEIS ===
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = Bot(token=TELEGRAM_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("chatbot-imobiliario")

# Modelo multilíngue (768 dimensões)
modelo = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# Link da pesquisa
LINK_PESQUISA = (
    "📝 *Participe da nossa pesquisa de satisfação:* "
    "[Clique aqui](https://docs.google.com/forms/d/e/1FAIpQLSeVz6AxoTvTB4pHxgKU-sso3GAUgA_irTBu4LGpkQgcTAThsQ/viewform?usp=header)"
)

PALAVRAS_ATENDENTE = [
    "atendente", "pessoa", "suporte", "falar com alguém", "ajuda humana", "representante"
]

app = Flask(__name__)

# === WEBHOOK DO TELEGRAM ===
@app.route("/", methods=["POST"])
def webhook():
    data = request.get_json()
    if not data or "message" not in data:
        return "ok"

    chat_id = data["message"]["chat"]["id"]
    pergunta = data["message"].get("text", "").lower().strip()
    print(f"📩 Mensagem recebida: {pergunta}")

    # 1️⃣ Atendimento humano
    if any(re.search(p, pergunta) for p in PALAVRAS_ATENDENTE):
        bot.send_message(
            chat_id,
            f"Certo! Vou encaminhar sua solicitação para um de nossos atendentes humanos. 👩‍💼\n\n{LINK_PESQUISA}",
            parse_mode="Markdown"
        )
        return "ok"

    # 2️⃣ Busca na base Pinecone
    try:
        vetor = modelo.encode(pergunta).tolist()
        resultados = index.query(vector=vetor, top_k=5, include_metadata=True)
        matches = [m for m in resultados.get("matches", []) if m["score"] >= 0.40]
        contexto = "\n\n".join([m["metadata"]["resposta"] for m in matches]) if matches else None
    except Exception as e:
        print(f"❌ Erro Pinecone: {e}")
        bot.send_message(chat_id, "Erro ao acessar base de conhecimento. Tente novamente mais tarde.")
        return "ok"

    # 3️⃣ Fallback inteligente
    if not contexto:
        respostas = {
            "casa": "Você gostaria de *comprar* ou *alugar* uma casa?",
            "imovel": "Você gostaria de *comprar* ou *alugar* um imóvel?",
            "comprar": "Você quer *comprar por financiamento* ou *à vista*?",
            "financiamento": "Posso te explicar sobre *financiamento imobiliário*! Quer saber as *condições* ou o *passo a passo*?",
            "documento": "Quer saber quais *documentos* são necessários para *comprar* ou *vender* um imóvel?",
            "programa": "Está se referindo aos *programas habitacionais*, como o *Minha Casa, Minha Vida*?"
        }
        for chave, resp in respostas.items():
            if chave in pergunta:
                bot.send_message(chat_id, f"{resp}\n\n{LINK_PESQUISA}", parse_mode="Markdown")
                return "ok"
        bot.send_message(chat_id, f"Desculpe, não encontrei essa informação. 🤝\n\n{LINK_PESQUISA}", parse_mode="Markdown")
        return "ok"

    # 4️⃣ Geração de resposta com OpenAI
    prompt = f"""
    Você é um assistente imobiliário especializado.
    Responda *somente* com base nas informações abaixo.

    --- CONTEXTO ---
    {contexto}

    --- PERGUNTA ---
    {pergunta}
    """

    try:
        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente imobiliário confiável, educado e direto."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        resposta_final = resposta.choices[0].message.content.strip()
        bot.send_message(chat_id, f"{resposta_final}\n\n{LINK_PESQUISA}", parse_mode="Markdown")
    except Exception as e:
        print(f"❌ Erro OpenAI: {e}")
        bot.send_message(chat_id, f"Desculpe, houve um erro ao gerar a resposta.\n\n{LINK_PESQUISA}", parse_mode="Markdown")

    return "ok"

# === VERIFICAÇÃO DE STATUS ===
@app.route("/", methods=["GET"])
def home():
    return "✅ Chatbot Imobiliário ativo na Vercel"

# === Exporta o app Flask para a Vercel ===
if __name__ != "__main__":
    # Necessário para Vercel reconhecer o objeto app
    app = app