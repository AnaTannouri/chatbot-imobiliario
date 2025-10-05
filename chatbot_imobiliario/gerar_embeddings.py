# gerar_embeddings.py
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from ler_base import carregar_base
import os

# === CONFIGURAÇÃO DO PINECONE ===
PINECONE_API_KEY = "pcsk_6iVoLb_8jecZ8RyPUn1fytrmP5JmWhsST699MppAqEsKpf7hLRfqMtUvc1kzcbB76p8ziS"
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "chatbot-imobiliario"

# === CRIA O ÍNDICE CASO NÃO EXISTA ===
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    print("🆕 Criando índice no Pinecone...")
    pc.create_index(
        name=index_name,
        dimension=768,  # compatível com o modelo multilíngue
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("✅ Índice criado com sucesso!")

index = pc.Index(index_name)

# === CARREGA TODAS AS BASES DE CONHECIMENTO ===
base_path = "base_conhecimento"
arquivos = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".txt")]

base_completa = []
for arquivo in arquivos:
    print(f"📂 Carregando base: {arquivo}")
    base_completa.extend(carregar_base(arquivo))

# === GERA EMBEDDINGS COM MODELO MULTILÍNGUE ===
print("🧠 Gerando embeddings com modelo multilíngue...")
modelo = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

perguntas = [item["pergunta"] for item in base_completa]
respostas = [item["resposta"] for item in base_completa]
embeddings = modelo.encode(perguntas, show_progress_bar=True).tolist()

# === ENVIA PARA O PINECONE EM LOTES ===
batch_size = 100
total = len(perguntas)
enviados = 0

print("🚀 Enviando embeddings para o Pinecone...")
for i in range(0, total, batch_size):
    batch = [
        {
            "id": f"{i+j}",
            "values": embeddings[i+j],
            "metadata": {"pergunta": perguntas[i+j], "resposta": respostas[i+j]}
        }
        for j in range(min(batch_size, total - i))
    ]

    index.upsert(vectors=batch)
    enviados += len(batch)
    print(f"✅ Lote {i // batch_size + 1} enviado ({len(batch)} vetores).")

print(f"\n🎉 Envio concluído! Total de {enviados} embeddings enviados para o Pinecone com sucesso.")
