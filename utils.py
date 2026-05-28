import re
import os
import io
import numpy as np
import torch
import torch.nn as nn
import joblib
from transformers import BertTokenizer, BertModel
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from huggingface_hub import hf_hub_download

REPO_MODELS = "Vocalin/intrinsec-plagiarism-detection-model"

MODELO_BERT      = "bert-base-uncased"
MAX_LEN          = 512
BATCH_SIZE       = 8
PALABRAS_VENTANA = 200
TRASLAPE         = 0.25
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM       = 256
DROPOUT          = 0.3

HF_TOKEN = os.getenv("HF_TOKEN", None)

class BertMLP(nn.Module):
    def __init__(self, bert, hidden_dim=256, n_classes=2, dropout=0.3):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def encode(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden = torch.stack(outputs.hidden_states[-4:], dim=0).sum(dim=0)

        mascara = attention_mask.clone()
        mascara[:, 0] = 0
        last_real = mascara.sum(dim=1, keepdim=True) - 1
        mascara.scatter_(1, last_real.clamp(min=0), 0)

        mascara_exp = mascara.unsqueeze(-1).float()
        suma_tokens = (hidden * mascara_exp).sum(dim=1)
        n_tokens    = mascara_exp.sum(dim=1).clamp(min=1e-9)
        return suma_tokens / n_tokens

    def forward(self, input_ids_A, attention_mask_A, token_type_ids_A,
                      input_ids_B, attention_mask_B, token_type_ids_B):
        emb_A = self.encode(input_ids_A, attention_mask_A, token_type_ids_A)
        emb_B = self.encode(input_ids_B, attention_mask_B, token_type_ids_B)
        return self.classifier(emb_A + emb_B)

# ─────────────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN 
# ─────────────────────────────────────────────────────────────────────────────
def inicializar_transformadores():
    """Carga la arquitectura BertMLP con los pesos fine-tuned y el SVM."""
    #Descargar modelos de hugging face
    bert_pt = hf_hub_download(
        repo_id=REPO_MODELS, 
        filename="mejor_modelo_bert_4.pt",
        token=HF_TOKEN
    )

    svm_trained = hf_hub_download(
        repo_id=REPO_MODELS, 
        filename="svm_embeddings_promedio_doc.joblib",
        token=HF_TOKEN
    )

    tokenizer = BertTokenizer.from_pretrained(MODELO_BERT)
    
    bert_base = BertModel.from_pretrained(MODELO_BERT)
    modelo_ft = BertMLP(bert_base, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)
    
    modelo_ft.load_state_dict(torch.load(bert_pt, map_location=DEVICE))
    modelo_ft.eval()
    
    # Cargar SVM
    svm = joblib.load(svm_trained)
    
    return tokenizer, modelo_ft, svm


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def segmentar_texto(texto, n_palabras, traslape):
    tokens = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', texto)]
    if not tokens:
        return []

    step = max(1, int(n_palabras * (1 - traslape)))
    segmentos = []

    i = 0
    while i + n_palabras <= len(tokens):
        ventana  = tokens[i:i + n_palabras]
        segmentos.append((texto[ventana[0][1]:ventana[-1][2]], ventana[0][1], ventana[-1][2], False))
        i += step

    if len(tokens) > n_palabras:
        ult = tokens[-n_palabras:]
        char_ini_ult, char_fin_ult = ult[0][1], ult[-1][2]

        if segmentos:
            overlap = max(0, min(char_fin_ult, segmentos[-1][2]) - max(char_ini_ult, segmentos[-1][1]))
            lon_ult = char_fin_ult - char_ini_ult
            es_cola = (overlap / lon_ult if lon_ult > 0 else 0) > 0.50
            if char_ini_ult != segmentos[-1][1]:
                segmentos.append((texto[char_ini_ult:char_fin_ult], char_ini_ult, char_fin_ult, es_cola))
        else:
            segmentos.append((texto[char_ini_ult:char_fin_ult], char_ini_ult, char_fin_ult, False))

    return segmentos

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS CON BERT FINE-TUNED
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def obtener_embeddings(textos, modelo, tokenizer):
    """Genera embeddings usando modelo.encode de BertMLP."""
    modelo.eval()
    todos = []

    for i in range(0, len(textos), BATCH_SIZE):
        batch = textos[i:i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        emb = modelo.encode(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
            enc["token_type_ids"].to(DEVICE),
        )
        todos.append(emb.cpu().numpy())

    return np.vstack(todos).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# RESOLUCIÓN, FUSIÓN Y GENERACIÓN DE REPORTES (PDF Y WEB)
# ─────────────────────────────────────────────────────────────────────────────
def resolver_y_fusionar_zonas(segmentos, labels, texto):
    if not segmentos:
        return b"", ""

    mascara_caracteres = np.full(len(texto), -1, dtype=np.int8)

    # Paso 1: Mapear Original (0)
    for seg, label in zip(segmentos, labels):
        if label == 0: 
            mascara_caracteres[seg[1]:seg[2]] = 0
            
    # Paso 2: Sobrescribir con Plagio (1) 
    for seg, label in zip(segmentos, labels):
        if label == 1: 
            mascara_caracteres[seg[1]:seg[2]] = 1

    # Preparar PDF en memoria con ReportLab
    pdf_en_memoria = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_en_memoria, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=50, bottomMargin=50
    )

    styles = getSampleStyleSheet()
    style_titulo = ParagraphStyle('TituloReporte', parent=styles['Heading1'], fontSize=22, leading=26, spaceAfter=15)
    style_cuerpo = ParagraphStyle('CuerpoReporte', parent=styles['Normal'], fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=10)

    historia = [
        Paragraph("Reporte de Detección de Plagio Intrínseco", style_titulo),
        Paragraph("<b>Código de colores:</b> Texto normal (Original) | <font backcolor='yellow'>Texto resaltado (Posible Plagio)</font>", styles['Normal']),
        Spacer(1, 20)
    ]

    def escapar_html(t):
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parrafos_raw = texto.split('\n')
    indice_char_actual = 0
    html_paginas_web = []

    for p_raw in parrafos_raw:
        if not p_raw.strip():
            historia.append(Spacer(1, 6))
            html_paginas_web.append("<br>")
            indice_char_actual += len(p_raw) + 1
            continue

        html_pdf = []
        html_web = []
        en_plagio = False

        for char in p_raw:
            label_char = mascara_caracteres[indice_char_actual] if indice_char_actual < len(mascara_caracteres) else -1
            es_plagio_actual = (label_char == 1)

            if es_plagio_actual and not en_plagio:
                html_pdf.append("<font backcolor='yellow'>")
                html_web.append("<mark class='resaltado-plagio'>")
                en_plagio = True
            elif not es_plagio_actual and en_plagio:
                html_pdf.append("</font>")
                html_web.append("</mark>")
                en_plagio = False

            texto_escapado = escapar_html(char)
            html_pdf.append(texto_escapado)
            html_web.append(texto_escapado)
            indice_char_actual += 1

        if en_plagio:
            html_pdf.append("</font>")
            html_web.append("</mark>")

        indice_char_actual += 1 
        
        historia.append(Paragraph("".join(html_pdf), style_cuerpo))
        html_paginas_web.append(f"<p>{''.join(html_web)}</p>")

    doc.build(historia)
    pdf_en_memoria.seek(0)
    
    return pdf_en_memoria.getvalue(), "".join(html_paginas_web)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE DE INFERENCIA DE MODELO
# ─────────────────────────────────────────────────────────────────────────────
def ejecutar_modelo(texto, svm, modelo_bert_ft, tokenizer):
    """Pipeline principal equivalente al main() del segundo script."""
    texto_limpio = texto.strip()
    
    segmentos = segmentar_texto(texto_limpio, PALABRAS_VENTANA, TRASLAPE)
    if len(segmentos) < 2:
        return b"", "<p>El texto es demasiado corto para ser analizado de forma intrínseca.</p>"
        
    textos_segs = [s[0] for s in segmentos]
    embs = obtener_embeddings(textos_segs, modelo_bert_ft, tokenizer)
    
    emb_promedio = embs.mean(axis=0)
    
    idx_activos  = [i for i, s in enumerate(segmentos) if not s[3]]
    segs_activos = [segmentos[i] for i in idx_activos]
    embs_activos = embs[idx_activos]
    
    X_inf  = embs_activos + emb_promedio
    labels = svm.predict(X_inf).tolist()
    
    return resolver_y_fusionar_zonas(segs_activos, labels, texto_limpio)