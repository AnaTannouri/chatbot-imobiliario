import os
from dotenv import load_dotenv
import re
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# === CARREGA VARIÁVEIS DO .env ===
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === CONFIGURAÇÕES ===
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("chatbot-imobiliario")

# 🧠 Modelo multilíngue — compatível com português
modelo = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# 🔗 Link da pesquisa de satisfação
LINK_PESQUISA = "📝 **Participe da nossa pesquisa de satisfação:** [Clique aqui](https://docs.google.com/forms/d/e/1FAIpQLSeVz6AxoTvTB4pHxgKU-sso3GAUgA_irTBu4LGpkQgcTAThsQ/viewform?usp=header)"

PALAVRAS_ATENDENTE = [
    "atendente", "pessoa", "suporte", "falar com alguém", "ajuda humana", "representante"
]

# === MENSAGEM INICIAL ===
MENSAGEM_BOAS_VINDAS = (
    "Olá! 👋 Sou o Chatbot Imobiliário!\n\n"
    "📘 Aviso de Privacidade (LGPD): As informações fornecidas serão tratadas conforme a "
    "Lei Geral de Proteção de Dados (Lei nº 13.709/2018), usadas apenas para fins de atendimento e consulta.\n\n"
    "Como posso lhe ajudar?\n\n"
    f"{LINK_PESQUISA}"
)

# === COMANDO /START ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(MENSAGEM_BOAS_VINDAS, parse_mode="Markdown")

# === RESPOSTAS ===
async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pergunta = update.message.text.lower().strip()
    print(f"📩 Mensagem recebida: {pergunta}")

    # 1️⃣ Atendimento humano
    if any(re.search(p, pergunta) for p in PALAVRAS_ATENDENTE):
        msg = "Certo! Vou encaminhar sua solicitação para um de nossos atendentes humanos. 👩‍💼"
        await update.message.reply_text(f"{msg}\n\n{LINK_PESQUISA}", parse_mode="Markdown")
        return

    # 2️⃣ Busca no Pinecone
    try:
        vetor = modelo.encode(pergunta).tolist()
        resultados = index.query(vector=vetor, top_k=5, include_metadata=True)
    except Exception as e:
        print(f"❌ Erro ao consultar o Pinecone: {e}")
        await update.message.reply_text("Desculpe, ocorreu um erro temporário. Tente novamente em alguns instantes.")
        return

    # 3️⃣ Filtra resultados relevantes
    matches = [m for m in resultados.get("matches", []) if m["score"] >= 0.40]
    contexto = "\n\n".join([m["metadata"]["resposta"] for m in matches]) if matches else None

    # 4️⃣ Fallback inteligente
    if not contexto:
        respostas = {
            "casa": "Você gostaria de **comprar** ou **alugar** uma casa?",
            "imovel": "Você gostaria de **comprar** ou **alugar** um imóvel?",
            "comprar": "Você quer **comprar um imóvel** por **financiamento** ou **à vista**?",
            "financiamento": "Posso te explicar sobre **financiamento imobiliário**! Quer saber as **condições** ou o **passo a passo**?",
            "documento": "Você quer saber quais **documentos** são necessários para **comprar** ou **vender** um imóvel?",
            "programa": "Está se referindo aos **programas habitacionais**, como o **Minha Casa, Minha Vida**?"
        }
        for chave, resp in respostas.items():
            if chave in pergunta:
                await update.message.reply_text(f"{resp}\n\n{LINK_PESQUISA}", parse_mode="Markdown")
                return
        await update.message.reply_text(
            f"Desculpe, não encontrei essa informação na base. 🤝\n\n{LINK_PESQUISA}",
            parse_mode="Markdown"
        )
        return

    # 5️⃣ Geração de resposta com base no RAG
    prompt = f"""
    Você é um assistente imobiliário especializado.
    Responda **somente** com base nas informações abaixo.

    --- CONTEXTO ---
    {contexto}

    --- PERGUNTA ---
    {pergunta}
    """

    try:
        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente imobiliário confiável, educado e objetivo."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        resposta_final = resposta.choices[0].message.content.strip()
        await update.message.reply_text(f"{resposta_final}\n\n{LINK_PESQUISA}", parse_mode="Markdown")

    except Exception as e:
        print(f"❌ Erro ao gerar resposta: {e}")
        await update.message.reply_text(f"Desculpe, houve um erro ao gerar a resposta.\n\n{LINK_PESQUISA}", parse_mode="Markdown")

# === INICIALIZAÇÃO ===
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))
    print("🤖 Chatbot Imobiliário conectado ao Telegram, Pinecone e OpenAI!")
    app.run_polling()

if __name__ == "__main__":
    main()
