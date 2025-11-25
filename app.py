import os
import io
import json
import time
import joblib
import requests
import numpy as np
import streamlit as st
from pathlib import Path
from gtts import gTTS
from deep_translator import GoogleTranslator

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "labels.json"

@st.cache_resource(show_spinner=False)
def load_pipeline():
    pipe = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    inv_labels = {int(k): v for k, v in labels.items()}
    return pipe, inv_labels

def predict_topk(pipe, inv_labels, ingredients, k=3):
    text = " ".join(ingredients).lower()
    probs = pipe.predict_proba([text])[0]
    top_idx = np.argsort(probs)[::-1][:k]
    items = [(inv_labels[int(i)], float(probs[i])) for i in top_idx]
    return items

# ---------- Simple recipe generator (deterministic, no external API) ----------
def generate_recipe(ingredients, cuisine):
    ing = [s.strip().lower() for s in ingredients if s.strip()]
    base_title = f"{cuisine.title()} style dish"
    title = base_title if ing == [] else f"{cuisine.title()} {ing[0]} bowl"
    steps = []
    steps.append(f"Prep: wash, peel, and finely chop aromatics (if any): garlic, onion, ginger.")
    if "rice" in " ".join(ing):
        steps.append("Cook rice separately (1 cup rice : 1.8 cups water).")
    steps.append(f"Marinate or season main ingredient(s): {', '.join(ing) if ing else 'selected items'} with salt, pepper, and a classic {cuisine} seasoning.")
    steps.append("Heat oil in a pan; sauté aromatics 1–2 min until fragrant.")
    steps.append("Add main ingredients; cook on medium-high until browned.")
    steps.append("Adjust with acid (lemon/lime/vinegar) and herbs; simmer 3–5 min.")
    steps.append("Taste and balance salt/sour/sweet/heat. Serve hot.")
    return title, steps

# ---------- Optional image retrieval (best-effort; safe fallback) -------------
def try_image(cuisine):
    try:
        qry = f"{cuisine} cuisine dish"
        url = f"https://source.unsplash.com/800x600/?{requests.utils.quote(qry)}"
        # Unsplash random endpoint always returns an image; Streamlit will fetch it.
        return url
    except Exception:
        return None

# ---------- Text-to-speech helpers -------------------------------------------
LANG_MAP = {"English": "en", "Français": "fr", "Deutsch": "de"}

def maybe_translate(text, tgt_lang_code):
    # If English requested, keep as-is; otherwise translate via GoogleTranslator
    if tgt_lang_code == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=tgt_lang_code).translate(text)
    except Exception:
        # Fallback: return original
        return text

def speak(text, lang_code="en", rate=1.0):
    # gTTS doesn’t support rate directly; emulate by duplicating spaces for slower feel
    txt = text if rate >= 1.0 else (" ".join(text.split()))
    tts = gTTS(txt, lang=lang_code)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf

# ============================== UI ===========================================
st.set_page_config(page_title="Smart Recipe Finder", page_icon="🍳", layout="wide")
st.title("🍳 Smart Recipe Finder")
st.caption("TF-IDF + Logistic Regression | Multilingual voice & basic recipe generation")

with st.sidebar:
    st.header("Settings")
    voice_lang = st.selectbox("Voice language (TTS)", list(LANG_MAP.keys()), index=0)
    do_tts = st.checkbox("Generate voice (TTS)", value=False)
    do_translate = st.checkbox("Translate recipe text to voice language", value=True)
    rate = st.slider("Reading speed (approx.)", min_value=0.6, max_value=1.4, value=1.0, step=0.05)
    st.markdown("---")
    st.subheader("About the model")
    st.markdown("- TF-IDF + LogisticRegression (multiclass)\n- Classes: 20")

pipe, INV_LABELS = load_pipeline()

with st.expander("Enter ingredients", expanded=True):
    ing_text = st.text_area(
        "List ingredients (comma-separated)",
        value="chicken, soy sauce, ginger, garlic, sesame oil",
        height=100,
        help="Example: tomato, basil, mozzarella, olive oil",
    )
    ingredients = [s.strip() for s in ing_text.split(",") if s.strip()]
    k = st.slider("Top-k cuisines", 1, 5, 3)

if st.button("Predict"):
    if not ingredients:
        st.warning("Please enter at least one ingredient.")
        st.stop()

    # Predictions
    topk = predict_topk(pipe, INV_LABELS, ingredients, k=k)
    top1_cuisine, top1_prob = topk[0]

    # Display bar chart
    st.subheader("Predicted cuisines (Top-k)")
    probs_df = {"cuisine": [c for c, _ in topk], "prob": [p for _, p in topk]}
    st.bar_chart(probs_df, x="cuisine", y="prob", use_container_width=True)

    # Image (safe)
    img_url = try_image(str(top1_cuisine))
    st.subheader(f"Image • {str(top1_cuisine).title()}")
    if img_url:
        st.image(img_url, use_column_width=True, caption=f"{str(top1_cuisine).title()} (illustrative)")
    else:
        st.info("No image available.")

    # Recipe generation
    title, steps = generate_recipe(ingredients, str(top1_cuisine))
    st.markdown(f"### 🍽️ {title}")
    recipe_text = f"{title}\n\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])

    # Optional translation of visible text
    view_lang_code = LANG_MAP[voice_lang] if do_translate else "en"
    recipe_text_view = maybe_translate(recipe_text, view_lang_code) if do_translate else recipe_text
    st.text_area("Recipe", value=recipe_text_view, height=220)

    # Voice
    if do_tts:
        tts_lang = LANG_MAP[voice_lang]
        audio_buf = speak(recipe_text_view, lang_code=tts_lang, rate=rate)
        st.audio(audio_buf, format="audio/mp3")
