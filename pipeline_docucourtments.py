import os
import json
import tempfile
import base64
import mimetypes
import re
from typing import List, Dict, Any, Optional

from PIL import Image
import imageio.v2 as imageio
import fitz  # pymupdf

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# ---------------------------
# A) Split document -> list of image paths
# ---------------------------

SUPPORTED_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"
}
SUPPORTED_PDF_EXTS = {".pdf"}


def split_document_to_image_paths(
    input_path: str,
    out_dir: Optional[str] = None,
    dpi: int = 200
) -> List[str]:
    """
    Input: pdf/tiff/gif/jpg/png...
    Output: list of image file paths (one per page).
    Saves images to out_dir (temp dir if None).
    """
    input_path = os.path.abspath(input_path)
    ext = os.path.splitext(input_path)[1].lower()

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="doc_pages_")
    os.makedirs(out_dir, exist_ok=True)

    paths = []
    base = os.path.splitext(os.path.basename(input_path))[0]

    # PDF -> render each page to PNG
    if ext in SUPPORTED_PDF_EXTS:
        doc = fitz.open(input_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            out_path = os.path.join(out_dir, f"{base}_page_{i:03d}.png")
            pix.save(out_path)
            paths.append(out_path)
        doc.close()
        return paths

    # Multi-frame images (TIFF/GIF)
    if ext in {".tif", ".tiff", ".gif"}:
        frames = imageio.mimread(input_path)
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            out_path = os.path.join(out_dir, f"{base}_page_{i:03d}.png")
            img.save(out_path)
            paths.append(out_path)
        return paths

    # Single images
    if ext in SUPPORTED_IMAGE_EXTS:
        out_path = os.path.join(out_dir, f"{base}_page_000.png")
        Image.open(input_path).save(out_path)
        return [out_path]

    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------
# B) Prompts
# ---------------------------

CLASSIFIER_PROMPT = """
Tu es un système de classification de documents.
Analyse l'image fournie et identifie le type de document.

Types possibles (choisis le plus proche):
- "etat_apurement"
- "recu_bancaire"
- "facture"
- "releve_bancaire"
- "quittance"
- "autre"

Important:
- Base-toi UNIQUEMENT sur ce que tu vois dans l'image.
- Si tu n'es pas sûr, choisis "autre" avec confiance faible.
- Notes: max 1 phrase.

Réponds UNIQUEMENT en JSON valide.
Format obligatoire:
{{
  "doc_type": "<un des types ci-dessus>",
  "confidence": <nombre entre 0 et 1>,
  "language": "<fr|ar|en|unknown>",
  "notes": "<court, optionnel>"
}}
"""

# Extracteur générique (docs simples)
EXTRACTOR_PROMPT_GENERIC = """
Tu es un extracteur d'information pour docuCourtments.
Type de document détecté : {doc_type}

Extrais les champs suivants si présents dans l'image :
- titre_no
- facture_no
- code_titre
- date
- banque
- reference
- nom_client

Extrais aussi TOUS les montants d'avances visibles.
Pour chaque avance :
- valeur (nombre)
- devise
- contexte court

IMPORTANT:
- Lis uniquement ce qu'il y a sur l'image.
- Ne devine jamais.
- Absent/illisible -> null.
- Si aucune avance -> liste vide.

Réponds UNIQUEMENT en JSON valide.
Format obligatoire:
{{
  "doc_type": "{doc_type}",
  "titre_no": null,
  "facture_no": null,
  "code_titre": null,
  "date": null,
  "banque": null,
  "reference": null,
  "nom_client": null,
  "avances": [
    {{
      "valeur": null,
      "devise": null,
      "contexte": null
    }}
  ]
}}
"""

# PASS 1 OCR fidèle (etat_apurement)
OCR_ITEMS_PROMPT = """
Tu es un système de lecture fidèle.
Recopie le contenu de l'image en séparant STRICTEMENT les items numérotés.

Format de sortie obligatoire (texte brut) :
1) <texte exact de l'item 1>
2) <texte exact de l'item 2>
3) <texte exact de l'item 3>
...

Ne résume pas. Ne reformule pas. Ne corrige pas les fautes.
Si une partie est illisible, mets [ILLISIBLE].
"""

# PASS 2 structure depuis texte (etat_apurement)
STRUCTURE_FROM_TEXT_PROMPT = """
Tu reçois le texte d'un ETAT D'APUREMENT item par item.

Texte :
{items_text}

Transformer ce texte en JSON STRICT selon le format ci-dessous.

Catégorie OBLIGATOIRE :
- si l’item contient "Avance" -> "avance"
- si l’item contient "surplus d'imputation" -> "surplus_imputation"
- si l’item contient "manque de rapatriement" -> "manque_rapatriement"
- si l’item contient "manque" -> "manque"
- sinon -> "autre"
Ne laisse JAMAIS categorie à null.

Règles :
- Ne mets un champ que s'il est dans l'item correspondant.
- Devise seulement si écrite, sinon null.
- Montant = nombre de l'item, ne mélange jamais avec numéro de titre.
- Extrais titres/factures si présents.
- Sous-lignes "-Titre N° ... = montant devise" -> reimputations.

Réponds UNIQUEMENT en JSON valide :
{{
  "doc_type": "etat_apurement",
  "date_document": null,
  "lieu": null,
  "service": null,
  "objet": null,
  "items": [
    {{
      "numero_item": null,
      "categorie": null,
      "montant": null,
      "devise": null,
      "titre_no": null,
      "titre_export_no": null,
      "facture_no": null,
      "annee_facture": null,
      "personne": null,
      "commentaire": null,
      "reimputations": [],
      "apurement_par": null
    }}
  ]
}}
"""

PROMPTS_BY_TYPE = {
    "etat_apurement": EXTRACTOR_PROMPT_GENERIC,  # non utilisé directement (2 passes)
    "recu_bancaire": EXTRACTOR_PROMPT_GENERIC,
    "facture": EXTRACTOR_PROMPT_GENERIC,
    "releve_bancaire": EXTRACTOR_PROMPT_GENERIC,
    "quittance": EXTRACTOR_PROMPT_GENERIC,
    "autre": EXTRACTOR_PROMPT_GENERIC
}


# ---------------------------
# C) LLM
# ---------------------------

def build_llm(model_name: str = "gemma3:latest"):
    return ChatOllama(model=model_name, temperature=0)


# ---------------------------
# D) Helpers
# ---------------------------

def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return json.loads(text)


def normalize_doc_type(doc_type: str) -> str:
    if not doc_type:
        return "autre"
    dt = doc_type.strip().lower()
    mapping = {
        "reçu bancaire": "recu_bancaire",
        "reçu_bancaire": "recu_bancaire",
        "recu bancaire": "recu_bancaire",
        "recu": "recu_bancaire",
        "etat d'apurement": "etat_apurement",
        "état d'apurement": "etat_apurement",
        "etat_apurement": "etat_apurement",
    }
    return mapping.get(dt, dt)


def image_path_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.replace(",", ".").replace(" ", ""))
        except:
            return None
    return None


def infer_categorie_from_text(text: str) -> str:
    if not text:
        return "autre"
    t = text.lower()

    if "avance" in t or "avances" in t:
        return "avance"
    if "surplus d'imputation" in t or "surplus d’imputation" in t or "surplus" in t:
        return "surplus_imputation"
    if "manque de rapatriement" in t:
        return "manque_rapatriement"
    if "manque" in t:
        return "manque"
    return "autre"


# ---------------------------
# E) Full pipeline
# ---------------------------

def process_document(
    input_path: str,
    model_name: str = "gemma3:latest",
    out_dir: Optional[str] = None
) -> Dict[str, Any]:

    llm = build_llm(model_name=model_name)
    page_paths = split_document_to_image_paths(input_path, out_dir=out_dir)

    results = []
    for path in page_paths:
        data_url = image_path_to_data_url(path)

        # ----- Prompt 1: classification -----
        cls_msgs = [
            SystemMessage(content=CLASSIFIER_PROMPT),
            HumanMessage(content=[
                {"type": "text",
                 "text": "Analyse cette page et retourne STRICTEMENT le JSON demandé."},
                {"type": "image_url", "image_url": data_url}
            ])
        ]

        cls_raw = llm.invoke(cls_msgs).content
        cls = safe_json_loads(cls_raw)

        doc_type = normalize_doc_type(cls.get("doc_type", "autre"))
        cls["doc_type"] = doc_type

        # ----- Prompt 2: extraction -----
        if doc_type == "etat_apurement":

            # PASS 1: OCR fidèle item par item
            ocr_msgs = [
                SystemMessage(content=OCR_ITEMS_PROMPT),
                HumanMessage(content=[
                    {"type": "text", "text": "Recopie fidèlement les items numérotés."},
                    {"type": "image_url", "image_url": data_url}
                ])
            ]
            items_text = llm.invoke(ocr_msgs).content

            # PASS 2: structuration depuis texte
            struct_prompt = STRUCTURE_FROM_TEXT_PROMPT.format(items_text=items_text)
            struct_msgs = [SystemMessage(content=struct_prompt)]
            ext_raw = llm.invoke(struct_msgs).content
            ext = safe_json_loads(ext_raw)

            # --- Fallback categories depuis items_text (robuste) ---
            item_text_map = {}
            current_num = None
            current_lines = []

            for line in items_text.splitlines():
                s = line.strip()
                m = re.match(r"^(\d{1,2})\s*[\)\.\-:]\s*(.*)$", s)
                if m:
                    if current_num is not None:
                        item_text_map[current_num] = " ".join(current_lines).strip()
                    current_num = m.group(1)
                    current_lines = [m.group(2).strip()]
                else:
                    if current_num is not None and s:
                        current_lines.append(s)

            if current_num is not None:
                item_text_map[current_num] = " ".join(current_lines).strip()

            for it in ext.get("items", []):
                if not isinstance(it, dict):
                    continue
                num = str(it.get("numero_item", "")).strip()
                raw_text = item_text_map.get(num, "")
                cat = it.get("categorie")
                if cat in [None, "", "null", "autre"]:
                    fallback_text = raw_text + " " + str(it.get("commentaire", "") or "")
                    it["categorie"] = infer_categorie_from_text(fallback_text)

        else:
            # extraction générique
            prompt_template = PROMPTS_BY_TYPE.get(doc_type, EXTRACTOR_PROMPT_GENERIC)
            extractor_prompt_filled = prompt_template.format(doc_type=doc_type)

            ext_msgs = [
                SystemMessage(content=extractor_prompt_filled),
                HumanMessage(content=[
                    {"type": "text",
                     "text": "Extrais les champs et retourne STRICTEMENT le JSON demandé."},
                    {"type": "image_url", "image_url": data_url}
                ])
            ]

            ext_raw = llm.invoke(ext_msgs).content
            ext = safe_json_loads(ext_raw)

        # ----- Post-processing -----

        # 1) avances (docs génériques)
        avances = ext.get("avances")
        if avances is None or not isinstance(avances, list):
            ext["avances"] = []
        else:
            for a in ext["avances"]:
                if isinstance(a, dict):
                    a["valeur"] = to_float(a.get("valeur"))
                    d = a.get("devise")
                    if isinstance(d, str) and "EURO" in d.upper():
                        a["devise"] = "EUR"

        # 2) items (etat_apurement) : conversions float + reimputations
        items = ext.get("items")
        if items is not None:
            if not isinstance(items, list):
                ext["items"] = []
            else:
                for it in ext["items"]:
                    if not isinstance(it, dict):
                        continue

                    it["montant"] = to_float(it.get("montant"))

                    d = it.get("devise")
                    if isinstance(d, str) and "EURO" in d.upper():
                        it["devise"] = "EUR"

                    # annee_facture string -> int
                    af = it.get("annee_facture")
                    if isinstance(af, str) and af.isdigit():
                        it["annee_facture"] = int(af)

                    # reimputations
                    reimp = it.get("reimputations")
                    if reimp is None or not isinstance(reimp, list):
                        it["reimputations"] = []
                    else:
                        for r in it["reimputations"]:
                            if isinstance(r, dict):
                                r["montant"] = to_float(r.get("montant"))
                                d2 = r.get("devise")
                                if isinstance(d2, str) and "EURO" in d2.upper():
                                    r["devise"] = "EUR"

        # 3) corrections spécifiques etat_apurement
        if ext.get("doc_type") == "etat_apurement":
            for it in ext.get("items", []):
                m = it.get("montant")
                if isinstance(m, (int, float)) and m > 200_000:
                    it["montant"] = None

        results.append({
            "page_path": path,
            "classification": cls,
            "extraction": ext
        })

    return {
        "input_path": os.path.abspath(input_path),
        "pages": results
    }


# ---------------------------
# F) CLI
# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to pdf/tiff/gif/jpg/png")
    parser.add_argument("--model", default="gemma3:latest")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    output = process_document(
        args.input_path,
        model_name=args.model,
        out_dir=args.out_dir
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
