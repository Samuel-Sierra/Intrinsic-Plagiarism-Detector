from contextlib import asynccontextmanager
from typing import Optional
import io
import base64
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import PyPDF2
from utils import modelo, inicializar_transformadores


recursos_globales = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Cargando modelos en memoria RAM (SVM + BERT)...")
        # Inicializa BERT y Tokenizer una sola vez
        tokenizer, bert_model, svm = inicializar_transformadores()
        
        recursos_globales["tokenizer"] = tokenizer
        recursos_globales["bert"] = bert_model
        recursos_globales["detector_plagio"] = svm
        print("¡Todos los modelos cargados con éxito!")
    except Exception as e:
        print(f"Error crítico al cargar los modelos: {e}")
        
    yield
    
    print("Limpiando recursos...")
    recursos_globales.clear()

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ══════════════════════════════════════════════════════════════════════════════
# VALIDACIONES
# ══════════════════════════════════════════════════════════════════════════════
def validate_text(text: str):
    length = len(text)
    if not (1000 <= length <= 180000):
        raise HTTPException(
            status_code=400, 
            detail=f"El texto debe tener entre 1,000 y 180,000 caracteres. (Actual: {length:,})"
        )
    # Validación simple sin regex pesado
    if not any(c.isalnum() for c in text):
        raise HTTPException(
            status_code=400, 
            detail="El documento no puede contener únicamente caracteres especiales o estar en blanco."
        )
    return True

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze-plagiarism/", response_class=HTMLResponse)
async def analyze_pdf(
    request: Request,
    file: Optional[UploadFile] = File(None), 
    textInput: Optional[str] = Form(None)
):
    texto_completo = ""
    try: 
        if not file and textInput:
            texto_completo = textInput.strip()
            
        elif file:
            if file.content_type != "application/pdf":
                raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")
                
            pdf_bytes_raw = await file.read()
            pdf_file = io.BytesIO(pdf_bytes_raw)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Reemplazo de bucle por comprensión eficiente
            texto_PDF = "".join([pagina.extract_text() or "" for pagina in pdf_reader.pages])
            texto_completo = texto_PDF.strip()
        else:
            raise HTTPException(status_code=400, detail="No se recibió ningún documento o texto.")

        validate_text(texto_completo)
        
        pdf_bytes_resultado, html_para_visor = modelo(
            texto_completo, 
            recursos_globales["detector_plagio"],
            recursos_globales["bert"],
            recursos_globales["tokenizer"]
        )
        
        base64_pdf = base64.b64encode(pdf_bytes_resultado).decode('utf-8')
        
        return templates.TemplateResponse("viewer.html", {
            "request": request,
            "texto_resaltado_html": html_para_visor,
            "pdf_base64": base64_pdf
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el análisis de plagio: {str(e)}")