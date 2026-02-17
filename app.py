import os
import time
import uuid
import base64
from pathlib import Path
from typing import Optional, Literal, List, Dict

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).parent
MEDIA_DIR = BASE_DIR / "media"
IMG_DIR = MEDIA_DIR / "images"
VID_DIR = MEDIA_DIR / "videos"
TEMPLATES_DIR = BASE_DIR / "templates"

IMG_DIR.mkdir(parents=True, exist_ok=True)
VID_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)

app = FastAPI(title="Estúdio Ultra Real IA (Imagem + Vídeo)")
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


# -----------------------------
# Utilitários
# -----------------------------
def now_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

def save_b64_to_png(b64_json: str, out_path: Path) -> None:
    raw = base64.b64decode(b64_json)
    out_path.write_bytes(raw)

def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY não configurada no Render. Vá em Settings → Environment e adicione OPENAI_API_KEY.",
        )
    return OpenAI(api_key=key)

def list_gallery() -> Dict[str, List[Dict[str, str]]]:
    images = sorted([p.name for p in IMG_DIR.glob("*.png")], reverse=True)
    videos = sorted([p.name for p in VID_DIR.glob("*.mp4")], reverse=True)
    return {
        "images": [{"name": n, "url": f"/media/images/{n}", "download": f"/download/images/{n}"} for n in images],
        "videos": [{"name": n, "url": f"/media/videos/{n}", "download": f"/download/videos/{n}"} for n in videos],
    }

def ultra_real_style() -> str:
    return (
        "fotografia hiper-realista, pele humana real com poros visíveis, microtextura, "
        "micro imperfeições naturais, sinais de expressão sutis, detalhes finos, "
        "iluminação natural/cinematográfica realista, lente 85mm, profundidade de campo suave, "
        "cores naturais, alta nitidez no rosto, sem aparência de CGI, sem cartoon, sem ilustração"
    )

def character_ref_file(character_id: str) -> Path:
    return IMG_DIR / f"{character_id}.ref.txt"

def set_character_ref(character_id: str, filename: str) -> None:
    character_ref_file(character_id).write_text(filename, encoding="utf-8")

def get_character_ref(character_id: str) -> Optional[str]:
    p = character_ref_file(character_id)
    if not p.exists():
        return None
    v = p.read_text(encoding="utf-8").strip()
    return v or None


# -----------------------------
# UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    tpl = env.get_template("index.html")
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    return tpl.render(has_key=has_key)


# -----------------------------
# Galeria / Download
# -----------------------------
@app.get("/api/galeria")
def api_galeria():
    return JSONResponse(list_gallery())

@app.get("/download/{kind}/{filename}")
def download(kind: str, filename: str):
    if kind not in ("images", "videos"):
        raise HTTPException(400, "Tipo inválido.")
    base = IMG_DIR if kind == "images" else VID_DIR
    path = base / filename
    if not path.exists():
        raise HTTPException(404, "Arquivo não encontrado.")
    return FileResponse(path, filename=filename)


# -----------------------------
# 1) Criar personagem (imagem base)
# -----------------------------
@app.post("/api/personagem/criar")
def criar_personagem(
    nome: str = Form(...),
    descricao: str = Form(...),
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = Form("1024x1024"),
):
    client = get_client()

    character_id = uuid.uuid4().hex

    prompt = (
        f"Crie um retrato fotográfico ultra-realista de um personagem IA fictício (não baseado em pessoa real). "
        f"Nome do personagem: {nome}. "
        f"Descrição: {descricao}. "
        f"Foco: rosto em destaque, identidade facial consistente, detalhes realistas da pele. "
        f"Estilo: {ultra_real_style()}."
        f"Fundo simples e limpo, sem texto, sem marca d'água."
    )

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size
    )

    b64 = result.data[0].b64_json
    filename = f"{now_id('personagem')}.png"
    out_path = IMG_DIR / filename
    save_b64_to_png(b64, out_path)

    set_character_ref(character_id, filename)

    return {"character_id": character_id, "image_name": filename, "image_url": f"/media/images/{filename}"}


# -----------------------------
# 2) Variação (mesmo rosto + novo ambiente)
# -----------------------------
@app.post("/api/personagem/variacao")
def gerar_variacao(
    character_id: str = Form(...),
    base_image: str = Form(...),  # nome do arquivo .png da galeria
    cena: str = Form(...),
    size: Literal["1024x1024", "1536x1024", "1024x1536"] = Form("1024x1024"),
):
    client = get_client()

    base_name = base_image.split("/")[-1]
    ref_path = IMG_DIR / base_name
    if not ref_path.exists():
        raise HTTPException(404, "Imagem base não encontrada na galeria.")

    prompt = (
        "Use a imagem enviada como referência do MESMO personagem (mesmo rosto). "
        f"Nova cena/ambiente: {cena}. "
        "Permita alterações de roupa/pose/cenário, mas mantenha a identidade facial consistente. "
        f"Estilo: {ultra_real_style()}. Sem texto, sem marca d'água."
    )

    # Tenta editar com imagem de referência (melhor consistência)
    try:
        with open(ref_path, "rb") as img:
            result = client.images.edit(
                model="gpt-image-1",
                image=img,
                prompt=prompt,
                size=size
            )
        b64 = result.data[0].b64_json
        method = "reference-image"
    except Exception:
        # fallback: só texto (pode variar mais o rosto)
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size
        )
        b64 = result.data[0].b64_json
        method = "text-fallback"

    filename = f"{now_id('variacao')}.png"
    out_path = IMG_DIR / filename
    save_b64_to_png(b64, out_path)

    return {"method": method, "image_name": filename, "image_url": f"/media/images/{filename}"}


# -----------------------------
# 3) Vídeo (rota pronta – depende do endpoint disponível na sua conta)
# -----------------------------
@app.post("/api/personagem/video")
def gerar_video(
    base_image: str = Form(...),
    prompt_video: str = Form(...),
):
    """
    Mantém a rota pronta. Se sua conta tiver acesso ao endpoint de vídeo, dá para ativar aqui.
    Se não tiver, retornamos um aviso amigável.
    """
    _ = get_client()  # só valida chave

    raise HTTPException(
        status_code=501,
        detail="Geração de vídeo: sua conta/endpoint ainda não está ativado neste app. "
               "Se você me disser qual provedor/modelo de vídeo você quer usar, eu integro (OpenAI se disponível, ou outro)."
    )