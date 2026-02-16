import os
import uuid
import base64
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Defina OPENAI_API_KEY nas variáveis de ambiente do Render (não no GitHub).")

client = OpenAI(api_key=API_KEY)

BASE_DIR = Path(__file__).parent
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(exist_ok=True)

TEMPLATES_DIR = BASE_DIR / "templates"
env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"])
)

app = FastAPI(title="Ultra Real AI - Personagem + Imagens + Vídeos")

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


# -----------------------------
# Utilitários
# -----------------------------
def save_b64_image_to_file(b64_json: str, out_path: Path) -> None:
    raw = base64.b64decode(b64_json)
    out_path.write_bytes(raw)

def id_file(ext: str) -> str:
    return f"{uuid.uuid4().hex}{ext}"

def ultra_real_prompt(base: str) -> str:
    # Observação: “poros e rugas” é totalmente permitido.
    # Só evite pedir para copiar rosto de pessoa real famosa/terceiro.
    return (
        base.strip()
        + "\n\nEstilo: fotografia hiper-realista, pele real com poros visíveis, microtextura, sinais de expressão sutis, "
          "iluminação natural, lente 85mm, profundidade de campo suave, alta nitidez no rosto, cores naturais."
    )

def safe_read_upload(file: UploadFile) -> bytes:
    content = file.file.read()
    if not content:
        raise HTTPException(400, "Arquivo vazio.")
    return content


# -----------------------------
# “Banco” simples (arquivo)
# -----------------------------
# Para demo: guardamos um arquivo TXT por personagem com o caminho da imagem de referência.
# Em produção: use SQLite ou Postgres.
def character_ref_path(character_id: str) -> Path:
    return MEDIA_DIR / f"{character_id}.ref.txt"

def set_character_ref(character_id: str, ref_image_filename: str) -> None:
    character_ref_path(character_id).write_text(ref_image_filename, encoding="utf-8")

def get_character_ref(character_id: str) -> Optional[str]:
    p = character_ref_path(character_id)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip() or None


# -----------------------------
# UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    tpl = env.get_template("index.html")

    # Lista imagens e vídeos gerados
    items = sorted([p.name for p in MEDIA_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4"}])
    return tpl.render(items=items)


# -----------------------------
# 1) Criar personagem (imagem-base)
# -----------------------------
@app.post("/api/character/create")
def create_character(
    description: str = Form(...),
    size: str = Form("1024x1024"),
):
    """
    Cria um personagem IA (imagem base) e gera um character_id.
    """
    character_id = uuid.uuid4().hex

    prompt = ultra_real_prompt(
        f"Crie um retrato ultra realista de um personagem original (não baseado em pessoa real). {description}"
    )

    # Geração de imagem (b64_json)
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size
    )

    b64_json = result.data[0].b64_json
    filename = id_file(".png")
    out_path = MEDIA_DIR / filename
    save_b64_image_to_file(b64_json, out_path)

    # Define essa imagem como referência do personagem
    set_character_ref(character_id, filename)

    return {
        "character_id": character_id,
        "ref_image": f"/media/{filename}",
        "download": f"/api/download/{filename}"
    }


# -----------------------------
# 2) Gerar nova imagem mantendo o mesmo rosto
# -----------------------------
@app.post("/api/character/{character_id}/image")
def generate_image_same_face(
    character_id: str,
    scene: str = Form(...),
    size: str = Form("1024x1024"),
):
    """
    Gera uma nova imagem usando a imagem do personagem como referência (image-to-image).
    Isso é o que ajuda a manter o mesmo rosto.
    """
    ref_filename = get_character_ref(character_id)
    if not ref_filename:
        raise HTTPException(404, "Personagem não encontrado. Crie o personagem primeiro.")

    ref_path = MEDIA_DIR / ref_filename
    if not ref_path.exists():
        raise HTTPException(500, "Referência do personagem não encontrada no servidor.")

    prompt = ultra_real_prompt(
        "Use a imagem fornecida como referência do mesmo personagem (mesmo rosto). "
        f"Crie esta nova cena: {scene}. "
        "Mantenha identidade facial consistente (mesmo formato do rosto, olhos, nariz e boca), "
        "mas permita mudanças de roupa, pose e ambiente."
    )

    # Tenta image-to-image (edits). Se o modelo/conta não suportar, cai para text-only (menos consistente).
    try:
        with open(ref_path, "rb") as img:
            result = client.images.edit(
                model="gpt-image-1",
                image=img,
                prompt=prompt,
                size=size
            )
        b64_json = result.data[0].b64_json
        method = "reference-image"
    except Exception:
        # fallback (pode variar mais o rosto)
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size
        )
        b64_json = result.data[0].b64_json
        method = "text-fallback"

    filename = id_file(".png")
    out_path = MEDIA_DIR / filename
    save_b64_image_to_file(b64_json, out_path)

    return {
        "character_id": character_id,
        "method": method,
        "image": f"/media/{filename}",
        "download": f"/api/download/{filename}"
    }


# -----------------------------
# 3) Gerar vídeo com o mesmo rosto (rota pronta)
# -----------------------------
@app.post("/api/character/{character_id}/video")
def generate_video_same_face(
    character_id: str,
    prompt: str = Form(...),
):
    """
    Vídeo depende de acesso ao endpoint de vídeo na sua conta.
    Mantemos a rota pronta. Se sua conta não tiver, retorna erro explicando.
    """
    ref_filename = get_character_ref(character_id)
    if not ref_filename:
        raise HTTPException(404, "Personagem não encontrado. Crie o personagem primeiro.")

    ref_path = MEDIA_DIR / ref_filename
    if not ref_path.exists():
        raise HTTPException(500, "Referência do personagem não encontrada no servidor.")

    # IMPORTANTE:
    # Dependendo do acesso da sua conta, o endpoint e assinatura podem mudar.
    # Por isso fazemos um try e damos uma mensagem clara.
    try:
        # Exemplo conceitual: usar a imagem como primeiro frame / referência.
        # Se sua conta suportar, adapte este trecho conforme o modelo/endpoints disponíveis.
        # (Se não suportar, dá exceção e a gente explica.)
        with open(ref_path, "rb") as img:
            video_job = client.videos.generate(  # pode não existir na sua SDK/conta
                model="sora",
                prompt=prompt,
                image=img
            )

        # Se retornar um URL/bytes, você salva em .mp4 dentro de /media.
        # Como isso varia por conta/modelo, deixamos a mensagem.
        return {"status": "ok", "detail": "Vídeo iniciado. (Ajuste o download conforme retorno do seu endpoint)."}
    except Exception as e:
        raise HTTPException(
            501,
            "Sua conta/SDK ainda não tem o endpoint de vídeo configurado aqui. "
            "O app já está pronto para imagens com rosto consistente. "
            "Se você quiser, eu adapto esta rota para o provedor de vídeo que você escolher (ou para o endpoint de vídeo que sua conta tiver). "
            f"Erro técnico: {type(e).__name__}"
        )


# -----------------------------
# Downloads
# -----------------------------
@app.get("/api/download/{filename}")
def download_file(filename: str):
    path = MEDIA_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Arquivo não encontrado.")
    return FileResponse(path, filename=filename)
  
