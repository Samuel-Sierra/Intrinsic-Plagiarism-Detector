import re
import io
import numpy as np
import torch
import joblib
from transformers import BertTokenizer, BertModel
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY

MODELO_BERT      = "bert-base-uncased"
MAX_LEN          = 512
BATCH_SIZE       = 8
PALABRAS_VENTANA = 250
TRASLAPE         = 0.25
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

def inicializar_transformadores():
    """Carga los modelos pesados una sola vez al iniciar el servidor."""
    tokenizer = BertTokenizer.from_pretrained(MODELO_BERT)
    bert_model = BertModel.from_pretrained(MODELO_BERT).to(DEVICE)
    svm = joblib.load("svm_embeddings_global.joblib")
    return tokenizer, bert_model, svm

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

@torch.no_grad()
def obtener_embeddings(textos, modelo, tokenizer):
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
        ).to(DEVICE)

        outputs = modelo(**enc, output_hidden_states=True)

        # Suma últimas 4 capas
        hidden = torch.stack(outputs.hidden_states[-4:], dim=0).sum(dim=0)

        # Mean pooling corregido y optimizado
        mascara = enc["attention_mask"].clone()
        mascara[:, 0] = 0
        last_real = mascara.sum(dim=1, keepdim=True) - 1
        mascara.scatter_(1, last_real.clamp(min=0), 0)

        mascara_exp = mascara.unsqueeze(-1).float()
        suma_tokens = (hidden * mascara_exp).sum(dim=1)
        n_tokens    = mascara_exp.sum(dim=1).clamp(min=1e-9)
        
        todos.append((suma_tokens / n_tokens).cpu().numpy())

    return np.vstack(todos).astype(np.float32)

def resolver_y_fusionar_zonas(segmentos, labels, texto):
    pdf_en_memoria = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_en_memoria, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=50, bottomMargin=50
    )

    if not segmentos:
        return b"", ""

    # Mapeo de caracteres eficiente
    mascara_caracteres = np.full(len(texto), -1, dtype=np.int8)
    for seg, label in zip(segmentos, labels):
        if label == 0: mascara_caracteres[seg[1]:seg[2]] = 0
    for seg, label in zip(segmentos, labels):
        if label == 1: mascara_caracteres[seg[1]:seg[2]] = 1

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
    
    # Lista para acumular los párrafos HTML destinados a viewer.html
    html_paginas_web = []

    for p_raw in parrafos_raw:
        if not p_raw.strip():
            historia.append(Spacer(1, 6))
            html_paginas_web.append("<br>") # Mantiene el espacio en la web
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
                html_web.append("<mark class='resaltado-plagio'>") # Clase CSS limpia para la web
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
        
        # Guardamos en la estructura del PDF y de la Web respectivamente
        historia.append(Paragraph("".join(html_pdf), style_cuerpo))
        html_paginas_web.append(f"<p>{''.join(html_web)}</p>")

    doc.build(historia)
    pdf_en_memoria.seek(0)
    
    # Retornamos ambos resultados analizados
    return pdf_en_memoria.getvalue(), "".join(html_paginas_web)


def modelo(texto, svm, bert, tokenizer):
    segmentos = segmentar_texto(texto, PALABRAS_VENTANA, TRASLAPE)
    if not segmentos:
        return b"", ""
        
    textos_segs = [s[0] for s in segmentos]
    embs        = obtener_embeddings(textos_segs, bert, tokenizer)
    emb_promedio = embs.mean(axis=0)
    
    idx_activos  = [i for i, s in enumerate(segmentos) if not s[3]]
    segs_activos = [segmentos[i] for i in idx_activos]
    embs_activos = embs[idx_activos]
    
    X_inf  = embs_activos + emb_promedio
    labels = svm.predict(X_inf).tolist()
    
    return resolver_y_fusionar_zonas(segs_activos, labels, texto)