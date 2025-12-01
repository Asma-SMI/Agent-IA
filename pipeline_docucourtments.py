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

import cv2
import numpy as np
import easyocr

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
    dpi: int = 300
) -> List[str]:
    input_path = os.path.abspath(input_path)
    ext = os.path.splitext(input_path)[1].lower()

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="doc_pages_")
    os.makedirs(out_dir, exist_ok=True)

    paths = []
    base = os.path.splitext(os.path.basename(input_path))[0]

    if ext in SUPPORTED_PDF_EXTS:
        doc = fitz.open(input_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            out_path = os.path.join(out_dir, f"{base}_page_{i:03d}.png")
            pix.save(out_path)
            paths.append(out_path)
        doc.close()
        return paths

    if ext in {".tif", ".tiff", ".gif"}:
        frames = imageio.mimread(input_path)
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            out_path = os.path.join(out_dir, f"{base}_page_{i:03d}.png")
            img.save(out_path)
            paths.append(out_path)
        return paths

    if ext in SUPPORTED_IMAGE_EXTS:
        out_path = os.path.join(out_dir, f"{base}_page_000.png")
        Image.open(input_path).save(out_path)
        return [out_path]

    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------
# B) Prompts
# ---------------------------

# Prompt classify
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
- Base-toi UNIQUEMENT sur ce que tu vois dans l'image ET le texte OCR si fourni.
- Si tu n'es pas sûr, choisis "autre" avec confiance faible.
- Notes: max 1 phrase.

Réponds UNIQUEMENT en JSON valide.
Format obligatoire:
{
  "doc_type": "<un des types ci-dessus>",
  "confidence": <nombre entre 0 et 1>,
  "language": "<fr|ar|en|unknown>",
  "notes": "<court, optionnel>"
}
"""

# Prompt OCR fidèle item-by-item
OCR_ITEMS_PROMPT = """
Tu es un système OCR fidèle. Tu dois COPIER le texte, pas l'interpréter.

Tâche:
- Recopie TOUTES les lignes visibles, sans aucune omission.
- Ne change aucun mot.
- Ne résume pas.
- Ne corrige pas l’orthographe.
- Ne fusionne pas deux items.

Format de sortie:
- Chaque item doit commencer par son numéro suivi de ')', par exemple: 1) , 2) , 3) ...
- Les lignes appartenant au même item doivent être recopiées juste en dessous.
- Mets UNE LIGNE VIDE entre deux items numérotés.
- Si un item contient des sous-lignes de détail / réimputation (ex: "Titre N° ... = montant"),
  alors ces sous-lignes doivent commencer par un tiret "-".
- Ne mets JAMAIS de balises ou placeholders comme <...> dans la sortie.

Si une partie est illisible:
- Recopie la ligne et remplace seulement la partie floue par [ILLISIBLE].

Important:
- Ne change JAMAIS "Avance" en "Surplus" ou l’inverse. Copie mot à mot.
"""

# Prompt structuration V2 (aligné sur tes champs finaux)
STRUCTURE_FROM_TEXT_PROMPT_V2 = """
Tu reçois le texte OCR d’un ETAT D’APUREMENT item par item.

Texte:
{items_text}

BUT: produire un JSON strict.

RÈGLES CRITIQUES:
1) DATE D’EN-TÊTE:
- La date en haut du document (ex: "Tunis, le 20/05/2025") est une date_document GLOBALE.
- Tu NE dois JAMAIS la mettre dans date_dom d’un item.
- date_dom peut être remplie UNIQUEMENT si une date est écrite DANS l’item
  et liée au titre / domiciliation. Sinon date_dom = null.

2) num_dom (N° TITRE):
- Extrais num_dom seulement si tu vois dans l’item un motif de titre:
  "Titre N°", "Titre No", "Titre N'", "Titre NP","/Titre N°",
  "Titre export N°", "Titre export No", "Titre export NP"
  suivi de chiffres (>=5).  
- num_dom = ces chiffres.  
- Si pas de motif titre -> null.
- Ne prends JAMAIS un montant comme num_dom.

3) facture:
- Extrais facture seulement si tu vois "Facture" ou "Fact" ou "FACT"
  suivi d’un numéro de type 058/2014, 23/14, 5/2014, etc.
- facture = "058/2014" ou "23/14" etc.
- Si absent -> null.

4) mnt_reglement / devise:
- mnt_reglement = montant principal de la phrase principale de l’item.
- devise = devise associée à ce montant principal (CHF/EUR/TND/USD).
- Ignore la devise des phrases "à apurer par ..." et des réimputations.

5) reimputations:
- Sous-lignes qui contiennent "Titre ... (Fact ..) = montant devise"
- Elles peuvent commencer par "-" OU directement par "Titre".
- Pour chaque réimputation, extrais:
  num_dom, facture, mnt_reglement, devise.

IMPORTANT:
- Ne devine rien.
- Si absent/illisible => null.

Réponds UNIQUEMENT en JSON valide:
{{
  "doc_type": "etat_apurement",
  "date_document": null,
  "items": [
    {{
      "numero_item": null,
      "ia_reg": null,
      "num_dom": null,
      "date_dom": null,
      "mnt_reglement": null,
      "devise": null,
      "date_reglement": null,
      "facture": null,
      "justificatif": null,
      "reimputations": []
    }}
  ]
}}
"""

# Extraction générique (si doc_type != etat_apurement)
EXTRACTOR_PROMPT_GENERIC = """
Tu es un extracteur d'information pour docuCourtments.
Type de document détecté : {doc_type}

Extrais les champs suivants si présents dans l'image OU le texte OCR :
- ia_reg (banque/institution : UIB, BIAT, BNA, SMI, ATB, etc.)
- num_dom (Titre N° / Titre export N°)
- date_dom (date liée au titre)
- mnt_reglement
- devise
- date_reglement
- justificatif (ex: reçu, swift, facture, etc.)

IMPORTANT:
- Lis uniquement ce qu'il y a sur l'image / OCR.
- Ne devine jamais. Flou/illisible -> null.
- Absent/illisible -> null.

Réponds UNIQUEMENT en JSON valide.
Format obligatoire:
{
  "doc_type": "{doc_type}",
  "ia_reg": null,
  "num_dom": null,
  "date_dom": null,
  "mnt_reglement": null,
  "devise": null,
  "date_reglement": null,
  "justificatif": null
}
"""

PROMPTS_BY_TYPE = {
    "etat_apurement": EXTRACTOR_PROMPT_GENERIC,
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
# D) Image enhancement + OCR
# ---------------------------

def enhance_image_for_ocr(img_path: str) -> str:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return img_path

    # débruitage léger
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # contraste doux
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # PAS de binarisation dure
    out_path = img_path.replace(".png", "_enh.png")
    cv2.imwrite(out_path, img)
    return out_path


_reader = None
def ocr_text(img_path: str) -> str:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['fr', 'en'], gpu=False)

    try:
        res = _reader.readtext(img_path, detail=0, paragraph=True)
        return "\n".join(res)
    except Exception:
        return ""


# ---------------------------
# E) Helpers
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


def clean_titre_no(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return x
    nums = re.findall(r"\d{5,}", x)
    return nums[0] if nums else None

def clean_facture_no(x: Any) -> Optional[str]:
    """
    Extrait un numéro de facture depuis une ligne OCR.
    Supporte :
      - Facture 058/2014
      - (Fact 24/14)
      - FACT 23/14
      - Facture N° 050/2014
      - 5/2014, 42/14, etc.
    Retourne format compact sans espaces, ex: "058/2014" ou "24/14".
    """
    if not isinstance(x, str):
        return None

    s = x.strip()

    # 1) motifs explicites Fact/FACT/Facture
    m = re.search(
        r"(?:facture|fact|FACT)\s*(?:n[°o]\s*)?[:#]?\s*(\d{1,4}\s*/\s*\d{2,4})",
        s,
        re.IGNORECASE
    )
    if m:
        return m.group(1).replace(" ", "")

    # 2) motif simple "058/2014" ou "23/14" n'importe où
    m2 = re.search(r"(\d{1,4}\s*/\s*\d{2,4})", s)
    if m2:
        return m2.group(1).replace(" ", "")

    # 3) fallback: un bloc de 2-4 chiffres seul (rare)
    m3 = re.search(r"\b\d{2,4}\b", s)
    return m3.group(0) if m3 else None


def extract_reimputations(raw_text: str) -> List[Dict[str, Any]]:
    if not isinstance(raw_text, str):
        return []

    reimps = []
    for line in raw_text.splitlines():
        l = line.strip()

        if not (l.startswith("-") or re.match(r"^titre", l, re.IGNORECASE)):
            continue

        titre_no = None
        mtitre = re.search(r"(titre(?:\s+export)?\s*n[°o]?\s*)(\d{5,})", l, re.IGNORECASE)
        if mtitre:
            titre_no = mtitre.group(2)

        facture_no = clean_facture_no(l)

        mmontant = re.search(r"=\s*([0-9\.\s]+(?:,[0-9]+)?)", l)
        montant = to_float(mmontant.group(1)) if mmontant else None

        mdev = re.search(r"\b(EUR|EUROS?|CHF|TND|USD)\b", l, re.IGNORECASE)
        devise = None
        if mdev:
            dv = mdev.group(1).upper()
            devise = "EUR" if "EURO" in dv else dv

        reimps.append({
            "num_dom": titre_no,
            "facture": facture_no,
            "mnt_reglement": montant,
            "devise": devise
        })

    return [r for r in reimps if any(v is not None for v in r.values())]


def normalize_items_text(ocr_txt: str) -> str:
    if not ocr_txt:
        return ocr_txt

    # 1) split grossier en lignes
    lines = [l.rstrip() for l in ocr_txt.splitlines()]

    out = []
    for l in lines:
        s = l.strip()
        if not s:
            continue

        # cas "1" seul => "1)"
        if re.fullmatch(r"\d{1,2}", s):
            s = s + ")"

        # cas "4 Un ..." => "4) Un ..."
        s = re.sub(r"^(\d{1,2})\s+(?=\S)", r"\1) ", s)

        # 2) ✅ split si un item apparaît au milieu de ligne: "... 6 Un ..."
        # on insère un retour ligne avant " 6) " ou " 6 "
        s = re.sub(r"\s+(\d{1,2})\s+(?=[A-Za-zÀ-ÿ])", r"\n\1) ", s)

        # 3) re-split si on a inséré des \n
        for part in s.split("\n"):
            part = part.strip()
            if not part:
                continue

            # ligne vide avant chaque nouvel item
            if re.match(r"^\d{1,2}\)", part):
                if out and out[-1] != "":
                    out.append("")
                out.append(part)
            else:
                out.append(part)

    return "\n".join(out).strip()

def looks_like_date(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    return bool(
        re.search(
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}",
            s
        )
    )


def fold_orphan_items_as_reimputations(ext: Dict[str, Any]) -> Dict[str, Any]:
    items = ext.get("items", [])
    if not items:
        return ext

    cleaned = []
    orphans = []

    for it in items:
        num_item = it.get("numero_item")
        # orphan = item sans numero_item mais qui ressemble à une réimputation
        if (num_item is None or str(num_item).strip() == "") and it.get("num_dom") and it.get("mnt_reglement"):
            orphans.append(it)
        else:
            cleaned.append(it)

    # on attache les orphans au DERNIER item réel (souvent item 8/9)
    if cleaned and orphans:
        parent = cleaned[-1]
        if "reimputations" not in parent or parent["reimputations"] is None:
            parent["reimputations"] = []

        for o in orphans:
            parent["reimputations"].append({
                "num_dom": clean_titre_no(o.get("num_dom")),
                "facture": clean_facture_no(o.get("facture")),
                "mnt_reglement": to_float(o.get("mnt_reglement")),
                "devise": o.get("devise")
            })

    ext["items"] = cleaned
    return ext

# ---------------------------
# F) Stage runners (prompt par prompt)
# ---------------------------

def run_classification(llm, data_url: str, ocr_txt: str) -> Dict[str, Any]:
    msgs = [
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=[
            {
                "type": "text",
                "text": f"Texte OCR (peut contenir erreurs):\n{ocr_txt}\n\n"
                        "Analyse l'image + OCR et retourne STRICTEMENT le JSON demandé."
            },
            {"type": "image_url", "image_url": data_url}
        ])
    ]
    raw = llm.invoke(msgs).content
    cls = safe_json_loads(raw)
    cls["doc_type"] = normalize_doc_type(cls.get("doc_type", "autre"))
    return cls


def run_items_ocr_only_easyocr(img_path: str, use_enhanced=True) -> str:
    # OCR sur image améliorée pour max recall
    path_for_ocr = enhance_image_for_ocr(img_path) if use_enhanced else img_path
    txt = ocr_text(path_for_ocr)

    # on renvoie le texte brut (sans tentative de structuration)
    return txt


def run_structure_items(llm, items_text: str) -> Dict[str, Any]:
    prompt = STRUCTURE_FROM_TEXT_PROMPT_V2.format(items_text=items_text)
    msgs = [SystemMessage(content=prompt)]
    raw = llm.invoke(msgs).content
    return safe_json_loads(raw)


def run_generic_extraction(llm, data_url: str, ocr_txt: str, doc_type: str) -> Dict[str, Any]:
    prompt = PROMPTS_BY_TYPE.get(doc_type, EXTRACTOR_PROMPT_GENERIC).format(doc_type=doc_type)
    msgs = [
        SystemMessage(content=prompt),
        HumanMessage(content=[
            {
                "type": "text",
                "text": f"Texte OCR (peut contenir erreurs):\n{ocr_txt}\n\n"
                        "Extrais les champs et retourne STRICTEMENT le JSON demandé."
            },
            {"type": "image_url", "image_url": data_url}
        ])
    ]
    raw = llm.invoke(msgs).content
    return safe_json_loads(raw)


# ---------------------------
# G) Post-processing final schema
# ---------------------------

# def post_process_etat_apurement(ext: Dict[str, Any], page_filename: str) -> Dict[str, Any]:
#     # récupère date d’en-tête si le LLM l’a mise
#     header_date = ext.get("date_document")

#     for it in ext.get("items", []):
#         if not isinstance(it, dict):
#             continue

#         # numero_item -> chiffres uniquement
#         if it.get("numero_item"):
#             m = re.search(r"\d{1,2}", str(it["numero_item"]))
#             it["numero_item"] = m.group(0) if m else it["numero_item"]

#         # num_dom nettoyage
#         it["num_dom"] = clean_titre_no(it.get("num_dom"))

#         # si num_dom trop court -> null
#         if it.get("num_dom") and len(str(it["num_dom"])) < 5:
#             it["num_dom"] = None

#         # montant float
#         it["mnt_reglement"] = to_float(it.get("mnt_reglement"))

#         # si num_dom == montant -> annuler num_dom
#         if it.get("num_dom") and it.get("mnt_reglement") is not None:
#             try:
#                 nd = str(it["num_dom"])
#                 mt = str(int(float(it["mnt_reglement"])))
#                 if nd == mt:
#                     it["num_dom"] = None
#             except:
#                 pass

#         # devise normalisée
#         d = it.get("devise")
#         if isinstance(d, str) and "EURO" in d.upper():
#             it["devise"] = "EUR"

#         # ✅ anti-fuite date en-tête
#         if header_date and it.get("date_dom") == header_date:
#             it["date_dom"] = None
#         # ou si date_dom == date du doc dans le texte
#         if it.get("date_dom") and re.search(r"\b20/05/2025\b", str(it["date_dom"])):
#             it["date_dom"] = None

#         # page
#         it["page"] = page_filename

#     return ext
def post_process_etat_apurement(ext: Dict[str, Any], page_filename: str) -> Dict[str, Any]:
    header_date = ext.get("date_document")

    for it in ext.get("items", []):
        if not isinstance(it, dict):
            continue

        # numero_item -> chiffres uniquement
        if it.get("numero_item"):
            m = re.search(r"\d{1,2}", str(it["numero_item"]))
            it["numero_item"] = m.group(0) if m else it["numero_item"]

        # num_dom nettoyage
        it["num_dom"] = clean_titre_no(it.get("num_dom"))

        if it.get("num_dom") and len(str(it["num_dom"])) < 5:
            it["num_dom"] = None

        it["mnt_reglement"] = to_float(it.get("mnt_reglement"))

        # devise normalisée
        d = it.get("devise")
        if isinstance(d, str) and "EURO" in d.upper():
            it["devise"] = "EUR"

        # anti-fuite date en-tête
        if header_date and it.get("date_dom") == header_date:
            it["date_dom"] = None
        if it.get("date_dom") and re.search(r"\b20/05/2025\b", str(it["date_dom"])):  # optionnel
            it["date_dom"] = None

        # ✅ facture : si c'est juste une année -> null
        if it.get("facture") and re.fullmatch(r"\d{4}", str(it["facture"]).strip()):
            it["facture"] = None

        # ✅ justificatif: tu ne veux PAS Avance/Surplus/Manque ici
        if it.get("justificatif"):
            it["justificatif"] = None

        it["page"] = page_filename

    return ext

# ---------------------------
# H) Full pipeline + staging
# ---------------------------

def process_document(
    input_path: str,
    model_name: str = "gemma3:latest",
    out_dir: Optional[str] = None,
    use_ocr: bool = True,
    stage: str = "full",
    page_index: Optional[int] = None
) -> Dict[str, Any]:

    llm = build_llm(model_name=model_name)
    _ = llm.invoke([HumanMessage(content="warmup")]).content

    page_paths = split_document_to_image_paths(input_path, out_dir=out_dir, dpi=400)
    if page_index is not None:
        page_paths = [page_paths[page_index]]

    results = []
    for path in page_paths:
        enh_path = enhance_image_for_ocr(path)
        ocr_txt = ocr_text(enh_path) if use_ocr else ""
        data_url = image_path_to_data_url(path)
        page_file = os.path.basename(path)

        # ====== STAGES ======
        if stage == "classify":
            cls = run_classification(llm, data_url, ocr_txt)
            results.append({"page": page_file, "classification": cls})
            continue

        if stage == "items_ocr":
            items_text = run_items_ocr_only_easyocr(path, use_enhanced=True)
            items_text = normalize_items_text(items_text)
            results.append({"page": page_file, "items_text": items_text})
            continue


        if stage == "structure_items":
            items_text = run_items_ocr_only_easyocr(path, use_enhanced=True)
            items_text = normalize_items_text(items_text)


            # 1) construire item_text_map
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

            # 2) structuration LLM
            ext = run_structure_items(llm, items_text)

            # forcer numero_item à partir de l'ordre réel
            real_nums = list(item_text_map.keys())  # ex: ["1","2","3"...]
            for i, it in enumerate(ext.get("items", [])):
                if i < len(real_nums):
                    it["numero_item"] = real_nums[i]

            # 3) ajouter reimputations par item
            for it in ext.get("items", []):
                if not isinstance(it, dict):
                    continue
                num = str(it.get("numero_item", "")).strip()
                raw_text = item_text_map.get(num, "")
                # ---- override num_dom & facture depuis texte brut (fiable) ----
                mt = re.search(r"(titre(?:\s+export)?\s*n[°o'Pp]?\s*)(\d{5,})", raw_text, re.IGNORECASE)
                if mt:
                    it["num_dom"] = mt.group(2)

                mf = re.search(
                    r"(?:facture|fact|FACT)\s*(?:n[°o]\s*)?[:#]?\s*(\d{1,4}\s*/\s*\d{2,4})",
                    raw_text,
                    re.IGNORECASE
                )
                if mf:
                    it["facture"] = mf.group(1).replace(" ", "")
                it["reimputations"] = extract_reimputations(raw_text)

            ext = fold_orphan_items_as_reimputations(ext)
            ext = post_process_etat_apurement(ext, page_file)

            # 4) post process final
            ext = post_process_etat_apurement(ext, page_file)

            results.append({"page": page_file, "extraction": ext})
            continue

        if stage == "extract_generic":
            cls = run_classification(llm, data_url, ocr_txt)
            doc_type = cls["doc_type"]
            ext = run_generic_extraction(llm, data_url, ocr_txt, doc_type)
            results.append({"page": page_file, "classification": cls, "extraction": ext})
            continue

        # ====== FULL ======
        if stage == "full":
            cls = run_classification(llm, data_url, ocr_txt)
            doc_type = cls["doc_type"]

            if doc_type == "etat_apurement":
                items_text = run_items_ocr(llm, data_url)
                items_text = normalize_items_text(items_text)


                # item_text_map (num_item -> texte brut)
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

                # structuration LLM
                ext = run_structure_items(llm, items_text)

                # ajout réimputations
                for it in ext.get("items", []):
                    if not isinstance(it, dict):
                        continue
                    num = str(it.get("numero_item", "")).strip()
                    raw_text = item_text_map.get(num, "")
                    it["reimputations"] = extract_reimputations(raw_text)

                # post-process final
                ext = post_process_etat_apurement(ext, page_file)

            else:
                ext = run_generic_extraction(llm, data_url, ocr_txt, doc_type)
                ext["page"] = page_file

            results.append({
                "page": page_file,
                "classification": cls,
                "extraction": ext
            })
            continue

        # si aucun stage ne matche
        raise ValueError(f"Unknown stage={stage}")

    # ✅ RETURN GLOBAL (sinon tu auras null)
    return {
        "input_path": os.path.abspath(input_path),
        "stage": stage,
        "pages": results
    }



# ---------------------------
# I) CLI
# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to pdf/tiff/gif/jpg/png")
    parser.add_argument("--model", default="gemma3:latest")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--no_ocr", action="store_true", help="Disable EasyOCR to save RAM")
    parser.add_argument(
        "--stage",
        default="full",
        choices=["full", "classify", "items_ocr", "structure_items", "extract_generic"],
        help="Run a single stage to debug prompts"
    )
    parser.add_argument("--page_index", type=int, default=None, help="Run only this page index (0-based)")
    args = parser.parse_args()

    output = process_document(
        args.input_path,
        model_name=args.model,
        out_dir=args.out_dir,
        use_ocr=not args.no_ocr,
        stage=args.stage,
        page_index=args.page_index
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
