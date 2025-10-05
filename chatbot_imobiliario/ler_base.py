# ler_base.py
def carregar_base(caminho_arquivo):
    base = []
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        linhas = f.readlines()

    pergunta, resposta = None, []
    for linha in linhas:
        linha = linha.strip()
        if linha.startswith("PERGUNTA:"):
            if pergunta and resposta:
                base.append({
                    "pergunta": pergunta,
                    "resposta": " ".join(resposta)
                })
            pergunta = linha.replace("PERGUNTA:", "").strip()
            resposta = []
        elif linha.startswith("RESPOSTA:"):
            resposta.append(linha.replace("RESPOSTA:", "").strip())
        elif linha:
            resposta.append(linha)
    if pergunta and resposta:
        base.append({
            "pergunta": pergunta,
            "resposta": " ".join(resposta)
        })
    return base
