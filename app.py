# app.py — Smart Recipe Finder (PRO, manual Meal Planner) — full Streamlit app
# ---------------------------------------------------------------------------------
# Features
# - TF–IDF + Logistic Regression model (joblib) → predict TOP-3 cuisines (distinct, with fallback)
# - One selection drives: preview, nutrition charts, TTS recipe, and HOST↔CHEF podcast
# - Robust TTS: Edge TTS (real male/female) → gTTS accents fallback (async-safe)
# - MP3 stitch via PyDub + ffmpeg (optional); download buttons for recipe/podcast MP3s
# - PDF export (recipe card with image + steps + macros) via reportlab (optional fallback)
# - Auto Shopping List (categorized) + export (TXT/CSV)
# - **Manual Meal Planner**: add/update/delete any day, free-text recipe names, re-order rows, CSV export
# - Optional translation (Deep-Translator → Google translate) with graceful fallback
#
# Suggested requirements.txt (you can omit ones you don’t need):
# streamlit
# numpy
# pandas
# scikit-learn
# joblib
# plotly
# edge-tts
# gTTS
# pydub
# ffmpeg-python
# reportlab
# deep-translator
#
# If deploying on Streamlit Cloud, add `packages.txt` with:
# ffmpeg
#
# Files needed in app directory:
# - cuisine_pipeline.joblib
# - labels.json   (either ["italian","mexican",...] or {"0":"italian","1":"mexican",...})

from __future__ import annotations
import io
import re
import json
import csv
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ==================== Optional deps (import lazily; degrade gracefully) ====================
def _has_edge_tts() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except Exception:
        return False

def _has_pydub_ffmpeg() -> bool:
    try:
        from pydub import AudioSegment  # noqa: F401
        return True
    except Exception:
        return False

def _has_reportlab() -> bool:
    try:
        import reportlab  # noqa: F401
        return True
    except Exception:
        return False

def _has_deep_translator() -> bool:
    try:
        import deep_translator  # noqa: F401
        return True
    except Exception:
        return False

EDGE_OK   = _has_edge_tts()
PYDUB_OK  = _has_pydub_ffmpeg()
PDF_OK    = _has_reportlab()
TRANS_OK  = _has_deep_translator()

# ================================= Paths / constants =================================
APP_DIR     = Path(__file__).resolve().parent
MODEL_PATH  = APP_DIR / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "labels.json"

TITLE = "Smart Recipe Finder (PRO)"
TOPK  = 3
np.random.seed(42)

# ============================ Persistent UI state (meal planner) =========================
if "meal_plan" not in st.session_state:
    # Stored as a list of dicts: {"Order": int, "Day": str, "Recipe": str}
    st.session_state["meal_plan"] = []

# ============================== Nutrition per 100 g (simple) =============================
NUTR_TABLE = {
    "chicken":{"kcal":165,"protein":31.0,"fat":3.6,"carbs":0.0},
    "beef":{"kcal":217,"protein":26.1,"fat":11.8,"carbs":0.0},
    "pork":{"kcal":242,"protein":27.0,"fat":14.0,"carbs":0.0},
    "tofu":{"kcal":76,"protein":8.0,"fat":4.8,"carbs":1.9},
    "lentil":{"kcal":116,"protein":9.0,"fat":0.4,"carbs":20.1},
    "bean":{"kcal":127,"protein":8.7,"fat":0.5,"carbs":22.8},
    "chickpea":{"kcal":164,"protein":8.9,"fat":2.6,"carbs":27.4},
    "garlic":{"kcal":149,"protein":6.4,"fat":0.5,"carbs":33.1},
    "ginger":{"kcal":80,"protein":1.8,"fat":0.8,"carbs":17.8},
    "onion":{"kcal":40,"protein":1.1,"fat":0.1,"carbs":9.3},
    "tomato":{"kcal":18,"protein":0.9,"fat":0.2,"carbs":3.9},
    "basil":{"kcal":23,"protein":3.2,"fat":0.6,"carbs":2.7},
    "olive oil":{"kcal":884,"protein":0.0,"fat":100.0,"carbs":0.0},
    "sesame oil":{"kcal":884,"protein":0.0,"fat":100.0,"carbs":0.0},
    "butter":{"kcal":717,"protein":0.9,"fat":81.1,"carbs":0.1},
    "soy sauce":{"kcal":53,"protein":8.0,"fat":0.6,"carbs":5.6},
    "rice":{"kcal":130,"protein":2.4,"fat":0.3,"carbs":28.0},
    "pasta":{"kcal":131,"protein":5.0,"fat":1.1,"carbs":25.0},
    "yogurt":{"kcal":59,"protein":10.0,"fat":0.4,"carbs":3.6},
}

SHOPPING_CATEGORIES = {
    "produce": ["garlic","ginger","onion","tomato","basil","spinach","kale","broccoli","parsley","cilantro","pepper","zucchini","lettuce","carrot","cucumber","lime","lemon"],
    "protein": ["chicken","beef","pork","tofu","egg","fish","shrimp","tuna","lentil","bean","chickpea","yogurt","paneer"],
    "dairy": ["milk","butter","cheese","cream","yogurt","ghee"],
    "pantry": ["rice","pasta","flour","bread","noodle","wheat","semolina","bulgur","couscous","soy sauce","olive oil","sesame oil","vinegar","stock","salt","pepper","sugar","gochujang","harissa"],
    "spices": ["cumin","chili","chilli","cayenne","turmeric","garam masala","paprika","oregano","thyme","rosemary","pepper flakes"],
}

# ================================= Cache model/labels =================================
@st.cache_resource(show_spinner="Loading model pipeline...")
def load_pipeline():
    if not MODEL_PATH.exists():
        st.error(f"Missing model at {MODEL_PATH}. Upload your TF–IDF+LogReg joblib file.")
        st.stop()
    if not LABELS_PATH.exists():
        st.error(f"Missing labels at {LABELS_PATH}. Provide a JSON list or {{idx: name}} mapping.")
        st.stop()
    pipe = joblib.load(MODEL_PATH)
    labels_raw = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    inv = {i: n for i, n in enumerate(labels_raw)} if isinstance(labels_raw, list) \
         else {int(k): v for k, v in labels_raw.items()}
    return pipe, inv

# ===================================== Helpers =====================================
def cuisine_image_url(cuisine: str) -> str:
    q = (cuisine + " plated dish").replace(" ", "%20")
    return f"https://source.unsplash.com/800x500/?{q}"

def parse_ingredients(text: str) -> List[str]:
    raw = [t.strip() for t in text.replace("؛", ",").replace("\n", ",").split(",")]
    return [t for t in raw if t]

def parse_mass_g(s: str, default: float = 100.0) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(g|gram|grams)\b", s.lower())
    return float(m.group(1)) if m else float(default)

def estimate_nutrition(items: List[str], default_mass_g: float = 100.0) -> Dict[str,float]:
    tot = {"kcal":0.0,"protein":0.0,"fat":0.0,"carbs":0.0}
    for it in items:
        it_l = it.lower()
        grams = parse_mass_g(it_l, default_mass_g)
        for key, nt in NUTR_TABLE.items():
            if key in it_l:
                mult = grams/100.0
                tot["kcal"]    += nt["kcal"]*mult
                tot["protein"] += nt["protein"]*mult
                tot["fat"]     += nt["fat"]*mult
                tot["carbs"]   += nt["carbs"]*mult
                break
    return tot

def qualitative_nutrition(ings: List[str]) -> Dict[str, float | str]:
    s = " ".join(ings).lower()
    hi_cal = sum(w in s for w in ["butter","oil","olive oil","ghee","cream","cheese","sugar","fried","bacon","nuts","peanut","sesame oil"])
    protein = sum(w in s for w in ["chicken","beef","pork","fish","tuna","egg","tofu","lentil","bean","chickpea","yogurt","paneer"])
    greens = sum(w in s for w in ["spinach","kale","broccoli","herb","parsley","cilantro","tomato","cucumber","lettuce","carrot","zucchini","pepper"])
    c_score = float(np.clip(0.20*hi_cal + 0.15*len(ings), 0.0, 1.0))
    p_score = float(np.clip(0.12*protein + 0.02*len(ings), 0.0, 1.0))
    h_score = float(np.clip(0.10*greens + 0.02*(len(ings)-hi_cal), 0.0, 1.0))
    cal_band = "low" if c_score <= 0.33 else ("medium" if c_score <= 0.66 else "high")
    return {"caloric_density":c_score,"protein_index":p_score,"healthiness":h_score,"calorie_band":cal_band}

def dietary_tags(ings: List[str]) -> List[str]:
    s = " ".join(ings).lower()
    tags = []
    animal = any(w in s for w in ["chicken","beef","pork","fish","shrimp","egg","yogurt","cheese","milk","butter","honey"])
    dairy  = any(w in s for w in ["cheese","milk","yogurt","cream","butter","ghee"])
    gluten = any(w in s for w in ["flour","bread","pasta","noodle","wheat","semolina","bulgur","couscous"])
    if not animal and not dairy: tags.append("vegan-ish")
    elif not animal: tags.append("vegetarian-ish")
    if not gluten: tags.append("gluten-light")
    if "pork" not in s and "bacon" not in s and "ham" not in s: tags.append("pork-free")
    if not any(w in s for w in ["wine","beer","ale","vodka","rum","whiskey","brandy"]): tags.append("no-alcohol")
    if any(w in s for w in ["chili","chilli","jalapeno","cayenne","gochujang","harissa","pepper flakes","hot","spicy"]): tags.append("spicy")
    return tags[:4]

def food_value_score(n: Dict[str,float]) -> float:
    return float(np.clip(0.35*n["protein_index"] + 0.40*n["healthiness"] - 0.25*n["caloric_density"], 0.0, 1.0))

# ================================== Recipes (deterministic) ==================================
FLAIR = {
    "indian":"Finish with garam masala and fresh cilantro.",
    "chinese":"Add a 1–1 splash of soy sauce and rice vinegar; sesame oil off-heat.",
    "italian":"Deglaze with a touch of white wine; finish with olive oil and basil.",
    "mexican":"Add cumin and chili powder; finish with lime and cilantro.",
    "japanese":"Season with mirin and soy; garnish with scallion.",
    "korean":"Stir in gochujang; top with sesame seeds.",
    "french":"Mount with a small knob of butter; finish with parsley/chives.",
    "thai":"Balance sweet–sour–salty with palm sugar, lime, fish sauce."
}
def _best_3(ings: List[str]) -> str:
    base = ", ".join([i.strip() for i in ings if i.strip()][:3])
    return base if base else "Seasonal Ingredients"

def generate_recipe(cuisine: str, ings: List[str]) -> str:
    title = f"{cuisine.title()}-Style Dish with {_best_3(ings).title()}"
    steps = [
        "Set up: wash and dry produce. Finely chop garlic, ginger, and onion; measure spices.",
        "Season mains lightly with salt and black pepper; keep at room temperature 10 minutes.",
        "Pan on medium heat. Add oil; bloom aromatics 60–90 seconds until fragrant (no browning).",
        "Add mains; sear 2–3 minutes to develop light color, then reduce heat to medium-low.",
        "Layer flavor: add core spices and a splash of stock or water. Simmer until tender.",
        "Adjust: a touch of acid (lemon/lime/vinegar) and fresh herbs for brightness.",
        "Rest 2 minutes; plate and finish with a drizzle of good oil or fresh herbs."
    ]
    steps.append(FLAIR.get(cuisine.lower(), "Finish with fresh herbs and a drizzle of olive oil."))
    body = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return f"{title}\n\n{body}"

# ================================== Prediction (TOP-3 robust) ==================================
def predict_topk(pipe, inv_labels: Dict[int,str], ings: List[str], k: int = TOPK) -> Tuple[List[str], np.ndarray]:
    text = " ".join(ings)
    proba = pipe.predict_proba([text])[0]
    order = np.argsort(proba)[::-1]
    names, values, seen = [], [], set()
    for idx in order:
        c = inv_labels[idx]
        if c not in seen:
            names.append(c); values.append(proba[idx]); seen.add(c)
        if len(names) == k:
            break
    if len(names) < k:
        FALLBACK = ["italian","mexican","indian","chinese","french","japanese","thai","korean","spanish","greek"]
        for c in FALLBACK:
            if len(names) == k: break
            if c not in seen:
                names.append(c); values.append(0.0); seen.add(c)
    return names, np.array(values, dtype=float)

# ================================== TTS utilities ==================================
EDGE_FEMALE_CHOICES = ["en-US-JennyNeural","en-GB-LibbyNeural","en-AU-NatashaNeural","en-CA-ClaraNeural"]
EDGE_MALE_CHOICES   = ["en-US-GuyNeural","en-GB-RyanNeural","en-AU-WilliamNeural","en-IN-PrabhatNeural"]
GTTs_ACCENTS = {"US": "com", "UK": "co.uk", "AU": "com.au", "CA": "ca", "IN": "co.in"}

def _ssml_wrap_chat(text: str, rate: str = "+0%", pitch: str = "+0%") -> str:
    safe = (text or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    return f"""<speak version="1.0" xml:lang="en-US">
  <mstts:express-as style="chat" styledegree="2" xmlns:mstts="https://www.w3.org/2001/mstts">
    <prosody rate="{rate}" pitch="{pitch}">{safe}</prosody>
  </mstts:express-as>
</speak>"""

async def _edge_ssml_to_bytes_async(ssml: str, voice: str) -> bytes:
    import edge_tts
    tts = edge_tts.Communicate(ssml=ssml, voice=voice)
    buf = io.BytesIO()
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()

def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

def tts_bytes_any(
    text: str,
    role: str,
    edge_voice_name: Optional[str],
    rate: str = "+0%",
    pitch: str = "+0%",
    gtts_tld: str = "com",
) -> Optional[bytes]:
    if edge_voice_name and EDGE_OK:
        try:
            ssml = _ssml_wrap_chat(text, rate=rate, pitch=pitch)
            return _run_async(_edge_ssml_to_bytes_async(ssml, edge_voice_name))
        except Exception:
            pass
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", tld=gtts_tld)
        out = io.BytesIO()
        tts.write_to_fp(out)
        return out.getvalue()
    except Exception:
        return None

def stitch_dialogue(
    dialogue: List[Tuple[str,str]],
    host_voice: Optional[str],
    chef_voice: Optional[str],
    pause_ms: int = 250,
    rate: str = "+0%",
    pitch: str = "+0%",
    tld_host: str = "com",
    tld_chef: str = "com"
) -> Optional[bytes]:
    if not PYDUB_OK:
        return None
    try:
        from pydub import AudioSegment
        track = AudioSegment.silent(duration=60)
        gap   = AudioSegment.silent(duration=max(0, int(pause_ms)))
        for role, text in dialogue:
            voice = host_voice if role == "HOST" else chef_voice
            tld   = tld_host if role == "HOST" else tld_chef
            b = tts_bytes_any(text, role, voice, rate=rate, pitch=pitch, gtts_tld=tld)
            if not b:
                return None
            seg = AudioSegment.from_file(io.BytesIO(b), format="mp3")
            track += seg + gap
        out = io.BytesIO()
        track.export(out, format="mp3")
        return out.getvalue()
    except Exception:
        return None

# ================================== Podcast dialogue ==================================
def _clean_recipe_lines(recipe_text: str) -> List[str]:
    lines = [ln.strip() for ln in recipe_text.split("\n") if ln.strip()]
    if lines and "-Style Dish" in lines[0]:
        lines = lines[1:]
    cleaned = [re.sub(r"^\d+\.\s*", "", ln) for ln in lines]
    return cleaned

def build_podcast_dialogue(
    host_name: str,
    chef_name: str,
    cuisine: str,
    ings: List[str],
    nutr: Dict[str,float|str],
    tags: List[str],
    recipe_text: str
) -> List[Tuple[str,str]]:
    base = ", ".join(ings[:2]) if ings else "basic pantry items"
    ttags = ", ".join(tags) if tags else "balanced"
    hcal  = nutr["calorie_band"]; pidx = f"{float(nutr['protein_index']):.2f}"; hidx = f"{float(nutr['healthiness']):.2f}"
    dlg: List[Tuple[str,str]] = [
        ("HOST", f"Hello everyone — {host_name} here, welcome to Flavor Talks!"),
        ("HOST", f"Today I’m joined by Chef {chef_name} to explore modern {cuisine.title()} flavors."),
        ("CHEF", f"Hi {host_name}, thanks for having me. I like this basket — {base} gives range."),
        ("HOST", f"In one line, what defines {cuisine.title()} cuisine?"),
        ("CHEF", f"It’s heritage and balance. Even with {base}, you taste its regional backbone."),
        ("HOST", "Quick nutrition snapshot?"),
        ("CHEF", f"Calorie density {hcal}; protein index {pidx}; healthiness {hidx}. Tags: {ttags}."),
        ("HOST", "Great — please walk us through the actual recipe, step by step."),
    ]
    steps = _clean_recipe_lines(recipe_text)
    step_no = 1
    for ln in steps:
        if ln:
            dlg.append(("CHEF", f"Step {step_no}: {ln}"))
            step_no += 1
    dlg.extend([
        ("HOST", "That was clean and practical. Thanks for cooking with us!"),
        ("CHEF", "My pleasure — a little acid at the end makes everything shine."),
    ])
    return dlg

# ================================== PDF generation ==================================
def generate_recipe_pdf(title: str, ingredients: List[str], recipe_text: str,
                        nutrition: Dict[str, float], image_url: str) -> Optional[bytes]:
    if not PDF_OK:
        return None
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        import requests

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
        story.append(Spacer(1, 12))

        # Image
        try:
            img_data = requests.get(image_url, timeout=10).content
            img = Image(io.BytesIO(img_data), width=400, height=250)
            story.append(img)
            story.append(Spacer(1, 12))
        except Exception:
            pass

        # Ingredients
        story.append(Paragraph("<b>Ingredients:</b>", styles["Heading2"]))
        for ing in ingredients:
            story.append(Paragraph(f"- {ing}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Instructions
        story.append(Paragraph("<b>Instructions:</b>", styles["Heading2"]))
        for line in recipe_text.split("\n"):
            if line.strip():
                story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 12))

        # Nutrition
        nutr_data = [
            ["Calories (kcal)", f"{nutrition.get('kcal',0):.1f}"],
            ["Protein (g)",     f"{nutrition.get('protein',0):.1f}"],
            ["Fat (g)",         f"{nutrition.get('fat',0):.1f}"],
            ["Carbs (g)",       f"{nutrition.get('carbs',0):.1f}"],
        ]
        table = Table(nutr_data)
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.black),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
            ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ]))
        story.append(Paragraph("<b>Nutrition (approx.):</b>", styles["Heading2"]))
        story.append(table)

        doc.build(story)
        pdf_value = buffer.getvalue()
        buffer.close()
        return pdf_value
    except Exception:
        return None

# ================================== Translation ==================================
LANG_CODES = {
    "English": "en", "Persian": "fa", "Turkish": "tr", "Arabic": "ar", "French": "fr", "Spanish": "es"
}

def translate_text(text: str, target_lang: str) -> str:
    """Try Deep-Translator (Google); fallback to original text on failure."""
    if not TRANS_OK or not text or target_lang == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="auto", target=target_lang)
        return translator.translate(text)
    except Exception:
        return text  # graceful fallback

# ================================== Shopping list ==================================
def categorize_shopping_list(ings: List[str]) -> Dict[str, List[str]]:
    cats: Dict[str, List[str]] = {k: [] for k in SHOPPING_CATEGORIES.keys()}
    cats["other"] = []
    for ing in ings:
        ing_l = ing.lower()
        matched = False
        for cat, keys in SHOPPING_CATEGORIES.items():
            if any(k in ing_l for k in keys):
                cats[cat].append(ing)
                matched = True
                break
        if not matched:
            cats["other"].append(ing)
    for k in cats:
        cats[k] = sorted(list(dict.fromkeys(cats[k])))  # unique and sorted
    return cats

def export_shopping_txt(cats: Dict[str, List[str]]) -> bytes:
    lines = []
    for cat, items in cats.items():
        if not items: continue
        lines.append(f"[{cat.UPPER()}]" if hasattr(cat, "UPPER") else f"[{cat.upper()}]")
        for it in items:
            lines.append(f"- {it}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")

def export_shopping_csv(cats: Dict[str, List[str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Category","Item"])
    for cat, items in cats.items():
        for it in items:
            writer.writerow([cat, it])
    return buffer.getvalue().encode("utf-8")

# ================================== UI ==================================
st.set_page_config(page_title=TITLE, page_icon="🍽️", layout="wide")
st.title(TITLE)
st.caption("Predict cuisines • 3 options • charts • TTS & podcast • PDF • shopping list • **manual meal planner** • translation")

# Sidebar — Audio
st.sidebar.header("Audio")
st.sidebar.markdown(f"- Edge TTS: **{'Yes' if EDGE_OK else 'No (gTTS fallback)**'}")
st.sidebar.markdown(f"- MP3 stitch (PyDub/ffmpeg): **{'Yes' if PYDUB_OK else 'No'}**")
if EDGE_OK:
    host_voice = st.sidebar.selectbox("Host voice (female)", EDGE_FEMALE_CHOICES, index=0, key="host_voice_sel")
    chef_voice = st.sidebar.selectbox("Chef voice (male)",   EDGE_MALE_CHOICES,   index=0, key="chef_voice_sel")
    tld_host = "com"; tld_chef = "com"
else:
    st.sidebar.info("Using gTTS accents (not truly gendered).")
    host_voice = None; chef_voice = None
    host_acc = st.sidebar.selectbox("Host accent (gTTS)", list(GTTs_ACCENTS.keys()), index=1, key="host_acc")
    chef_acc = st.sidebar.selectbox("Chef accent (gTTS)", list(GTTs_ACCENTS.keys()), index=0, key="chef_acc")
    tld_host = GTTs_ACCENTS[host_acc]; tld_chef = GTTs_ACCENTS[chef_acc]

voice_rate        = st.sidebar.selectbox("Voice speed", ["-10%","-5%","+0%","+5%","+10%"], index=2, key="rate_sel")
voice_pitch       = st.sidebar.selectbox("Voice pitch", ["-2%","+0%","+2%","+4%"], index=1, key="pitch_sel")
enable_recipe_tts = st.sidebar.checkbox("Enable voice for selected recipe", value=True, key="rec_tts_chk")
enable_podcast    = st.sidebar.checkbox("Enable conversational podcast (Host ↔ Chef)", value=True, key="podcast_chk")
podcast_pause     = st.sidebar.slider("Pause between turns (ms)", 150, 800, 300, step=50, key="pause_sel")
host_name         = st.sidebar.text_input("Host display name", value="Sara", key="host_name_in")
chef_name         = st.sidebar.text_input("Chef display name", value="Masoud", key="chef_name_in")

# Sidebar — Export & Planner
st.sidebar.header("Export & Planner")
target_lang_name = st.sidebar.selectbox("Translate to", list(LANG_CODES.keys()), index=0)
meal_days        = st.sidebar.slider("Default day index (for quick pick)", 1, 7, 3, key="days_sel")

with st.sidebar.expander("About the model", expanded=False):
    st.markdown("- Logistic Regression over TF–IDF\n- Trained on Yummly ‘What’s Cooking?’ dataset")

# Load model/labels
pipe, INV = load_pipeline()
st.sidebar.markdown(f"- Classes: **{len(INV)}**")

# Layout
left, right = st.columns([1.35, 1.0])

# ================================== INPUT PANEL ==================================
with left:
    st.subheader("Ingredients")
    demo = "chicken, soy sauce, ginger, garlic, sesame oil"
    ing_text = st.text_area(
        "Comma-separated or one per line",
        value=demo, height=120,
        placeholder="e.g., tomato, basil, garlic, olive oil",
        key="ing_text"
    )
    ings = parse_ingredients(ing_text)

    run = st.button("Predict cuisines & build 3 recipe options", type="primary", use_container_width=True, key="predict_btn")
    if run:
        if not ings:
            st.warning("Please provide at least one ingredient.")
            st.stop()

        cuisines, probs = predict_topk(pipe, INV, ings, k=TOPK)
        df_pred = pd.DataFrame({"cuisine": cuisines, "probability": probs})

        options_meta: Dict[str, Dict] = {}
        for c in cuisines:
            recipe = generate_recipe(c, ings)
            macro  = estimate_nutrition(ings, default_mass_g=100.0)
            nutr   = qualitative_nutrition(ings)
            fvs    = food_value_score(nutr)
            options_meta[c] = {"recipe": recipe, "macro": macro, "nutr": nutr, "fvs": fvs}

        st.session_state["pred_ready"]       = True
        st.session_state["df_pred"]          = df_pred
        st.session_state["cuisines"]         = cuisines
        st.session_state["options_meta"]     = options_meta
        st.session_state["selected_cuisine"] = cuisines[0]
        st.session_state["ings"]             = ings
        st.rerun()

# ================================== RENDER PANEL ==================================
with left:
    if st.session_state.get("pred_ready", False):
        df_pred      = st.session_state["df_pred"]
        cuisines     = st.session_state["cuisines"]
        options_meta = st.session_state["options_meta"]
        selected     = st.session_state.get("selected_cuisine", cuisines[0])
        ings         = st.session_state.get("ings", [])

        # Predictions chart
        st.markdown("### Top predictions")
        fig_pred = go.Figure(data=[go.Bar(x=df_pred["cuisine"], y=df_pred["probability"])])
        fig_pred.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                               yaxis=dict(title="Probability", rangemode="tozero"),
                               xaxis=dict(title="Cuisine"), height=300, template="simple_white")
        st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False}, key="pred_chart")

        # Radio selection
        st.markdown("### Explore three recipe options")
        chosen = st.radio(
            "Pick one to preview & voice:",
            options=cuisines,
            index=cuisines.index(selected),
            horizontal=True,
            label_visibility="collapsed",
            key="recipe_radio",
        )
        if chosen != selected:
            st.session_state["selected_cuisine"] = chosen
            selected = chosen

        # Preview one
        c = selected
        st.markdown(f"**Cuisine:** {c.title()}")
        st.image(cuisine_image_url(c), use_column_width=True)
        recipe_text = options_meta[c]["recipe"]

        # TRANSLATION (text only; audio uses translated text if target ≠ en)
        tgt_code = LANG_CODES.get(target_lang_name, "en")
        display_recipe_text = translate_text(recipe_text, tgt_code)

        st.text_area("Recipe", value=display_recipe_text, height=260, label_visibility="collapsed", key=f"recipe_preview_{c}")

        # Macro chart
        m = options_meta[c]["macro"]
        macro_names = ["Calories (kcal)", "Protein (g)", "Fat (g)", "Carbs (g)"]
        macro_vals  = [m["kcal"], m["protein"], m["fat"], m["carbs"]]
        fig_macro = go.Figure(data=[go.Bar(x=macro_names, y=macro_vals)])
        fig_macro.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                yaxis=dict(title="Amount", rangemode="tozero"),
                                xaxis=dict(title=""), height=300, template="simple_white")
        st.plotly_chart(fig_macro, use_container_width=True, config={"displayModeBar": False}, key=f"macro_{c}")

        # Value chart
        n = options_meta[c]["nutr"]
        value_names = ["Caloric density", "Protein index", "Healthiness", "Food Value Score"]
        value_vals  = [n["caloric_density"], n["protein_index"], n["healthiness"], options_meta[c]["fvs"]]
        fig_val = go.Figure(data=[go.Bar(x=value_names, y=value_vals)])
        fig_val.update_yaxes(range=[0, 1])
        fig_val.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                              yaxis=dict(title="Score (0–1)", rangemode="tozero"),
                              xaxis=dict(title=""), height=300, template="simple_white")
        st.plotly_chart(fig_val, use_container_width=True, config={"displayModeBar": False}, key=f"value_{c}")

        # ========================== VOICE: Recipe TTS ==========================
        if enable_recipe_tts:
            st.markdown("#### 🔊 Voice recipe")
            text_for_audio = display_recipe_text  # already translated if requested
            tld_for_host = tld_host if not EDGE_OK else "com"
            audio = tts_bytes_any(text_for_audio, role="HOST", edge_voice_name=host_voice,
                                  rate=voice_rate, pitch=voice_pitch, gtts_tld=tld_for_host)
            if audio:
                st.audio(audio, format="audio/mp3")
                st.download_button("Download recipe MP3", data=audio,
                                   file_name=f"{c}_recipe.mp3", mime="audio/mpeg", key=f"dl_recipe_{c}")
            else:
                st.info("TTS unavailable (Edge blocked and/or gTTS missing).")

        # ========================== PODCAST (Dialogue) =========================
        if enable_podcast:
            st.markdown("#### 🎙️ Conversational podcast (Host ↔ Chef)")
            dlg_en = build_podcast_dialogue(host_name, chef_name, c, ings, n, dietary_tags(ings), recipe_text)
            dlg = [(r, translate_text(t, tgt_code)) for (r, t) in dlg_en] if tgt_code != "en" else dlg_en
            st.markdown("\n".join([f"**{r}:** {t}" for r, t in dlg]))

            stitched = stitch_dialogue(
                dlg, host_voice, chef_voice,
                pause_ms=int(podcast_pause), rate=voice_rate, pitch=voice_pitch,
                tld_host=tld_host, tld_chef=tld_chef
            )
            if stitched:
                st.audio(stitched, format="audio/mp3")
                st.download_button("Download podcast MP3", data=stitched,
                                   file_name=f"{c}_podcast.mp3", mime="audio/mpeg", key=f"dl_podcast_{c}")
            else:
                st.caption("Single-file stitch unavailable — playing turn-by-turn.")
                for i, (role, text) in enumerate(dlg, 1):
                    vname = host_voice if role == "HOST" else chef_voice
                    tld   = tld_host if role == "HOST" else tld_chef
                    b = tts_bytes_any(text, role, vname, rate=voice_rate, pitch=voice_pitch, gtts_tld=tld)
                    if b:
                        st.markdown(f"*Turn {i} — {role}*")
                        st.audio(b, format="audio/mp3")
                    else:
                        st.info("TTS unavailable in this environment.")
                        break

        # ========================== EXPORT: PDF Recipe =========================
        st.markdown("#### 📄 Export")
        if PDF_OK:
            pdf_bytes = generate_recipe_pdf(
                title=f"{c.title()}-Style Dish",
                ingredients=ings,
                recipe_text=display_recipe_text,
                nutrition=m,
                image_url=cuisine_image_url(c),
            )
            if pdf_bytes:
                st.download_button("Download Recipe PDF", data=pdf_bytes,
                                   file_name=f"{c}_recipe.pdf", mime="application/pdf", key=f"dl_pdf_{c}")
        else:
            st.info("Install `reportlab` to enable PDF export.")

        # ========================== SHOPPING LIST =============================
        st.markdown("#### 🧾 Shopping list")
        cats = categorize_shopping_list(ings)
        with st.expander("View categorized list", expanded=False):
            for cat, items in cats.items():
                if not items: continue
                st.markdown(f"**{cat.title()}**: " + ", ".join(items))
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("Download TXT", data=export_shopping_txt(cats),
                               file_name=f"{c}_shopping_list.txt", mime="text/plain")
        with col_b:
            st.download_button("Download CSV", data=export_shopping_csv(cats),
                               file_name=f"{c}_shopping_list.csv", mime="text/csv")

        # ========================== MEAL PLANNER (MANUAL) =============================
        st.markdown("#### 🍽️ Meal planner (manual)")

        # Day choices (customize if you prefer weekdays)
        day_choices = [f"Day {i}" for i in range(1, 8)]
        default_day_index = min(len(day_choices)-1, max(0, meal_days-1))

        col_day, col_recipe, col_custom = st.columns([1, 1, 1.2])
        with col_day:
            sel_day = st.selectbox("Select a day", day_choices, index=default_day_index, key="plan_sel_day")

        with col_recipe:
            sel_recipe_from_pred = st.selectbox("Pick recipe (from predictions)", cuisines, index=0, key="plan_sel_recipe")

        with col_custom:
            custom_recipe = st.text_input("Or type a custom recipe name", value="", key="plan_custom_recipe")

        final_recipe_name = (custom_recipe.strip() or sel_recipe_from_pred).strip()

        col_add, col_delete, col_clear = st.columns([1,1,1])
        with col_add:
            if st.button("Add / Update day", use_container_width=True):
                found = False
                for row in st.session_state["meal_plan"]:
                    if row["Day"] == sel_day:
                        row["Recipe"] = final_recipe_name
                        found = True
                        break
                if not found:
                    next_order = (max([r["Order"] for r in st.session_state["meal_plan"]] or [0]) + 1)
                    st.session_state["meal_plan"].append({"Order": next_order, "Day": sel_day, "Recipe": final_recipe_name})

        with col_delete:
            if st.button("Delete selected day", use_container_width=True):
                st.session_state["meal_plan"] = [r for r in st.session_state["meal_plan"] if r["Day"] != sel_day]

        with col_clear:
            if st.button("Clear all", use_container_width=True):
                st.session_state["meal_plan"] = []

        if st.session_state["meal_plan"]:
            plan_df = pd.DataFrame(st.session_state["meal_plan"]).sort_values("Order").reset_index(drop=True)
            st.caption("Tip: Change the **Order** numbers to re-arrange rows, then click **Apply changes**.")
            edited_df = st.data_editor(
                plan_df[["Order","Day","Recipe"]],
                num_rows="dynamic",
                use_container_width=True,
                key="plan_editor",
            )

            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("Apply changes", use_container_width=True):
                    cleaned = []
                    for _, row in edited_df.iterrows():
                        try:
                            ord_val = int(row["Order"])
                        except Exception:
                            continue
                        day_val = str(row["Day"]).strip()
                        rec_val = str(row["Recipe"]).strip()
                        if day_val and rec_val:
                            cleaned.append({"Order": ord_val, "Day": day_val, "Recipe": rec_val})
                    dedup = {(r["Order"], r["Day"]): r for r in cleaned}
                    st.session_state["meal_plan"] = sorted(list(dedup.values()), key=lambda x: x["Order"])
                    st.success("Meal plan updated.")

            with c2:
                if st.session_state["meal_plan"]:
                    export_df = pd.DataFrame(st.session_state["meal_plan"]).sort_values("Order")
                    st.download_button(
                        "Download plan CSV",
                        data=export_df.to_csv(index=False).encode("utf-8"),
                        file_name="meal_plan.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        else:
            st.info("No items yet. Select a day + recipe (or type custom) and click **Add / Update day**.")

# =========================== RIGHT PANE: HOW-TO / NOTES =======================
with right:
    st.subheader("How to use")
    st.markdown(
        "1) Enter ingredients\n\n"
        "2) Click **Predict cuisines & build 3 recipe options**\n\n"
        "3) Use the **radio** to pick one cuisine (selection syncs)\n\n"
        "4) Toggle **Voice**/**Podcast**, pick voices, set speed/pitch\n\n"
        "5) Translate text (sidebar) and export **PDF/MP3**\n\n"
        "6) Build the **manual meal plan**: choose day, pick/enter recipe, add/update, delete, and re-order; then download CSV"
    )
    st.markdown("---")
    st.subheader("Notes")
    st.markdown(
        "- Provide **cuisine_pipeline.joblib** and **labels.json** in the app folder.\n"
        "- For real female/male voices, Edge TTS must be available. Otherwise, gTTS accents are used.\n"
        "- MP3 stitching requires **pydub** and **ffmpeg**. PDF export requires **reportlab**.\n"
        "- Translation uses **deep-translator** (Google). If unavailable or failing, original English is used."
    )

st.markdown("---")
st.caption("Smart Recipe Finder (PRO) • cooking coach • shopping helper • manual meal planner")
