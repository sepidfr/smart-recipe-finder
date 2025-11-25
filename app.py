import os
import io
import json
import joblib
import requests
import numpy as np
import streamlit as st
from pathlib import Path
from gtts import gTTS

# ---------------------------------------------------------
# PATHS (match your GitHub repo — files in same folder)
# ---------------------------------------------------------
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "cuisine_pipeline.joblib"
LABELS_PATH = APP_DIR / "labels.json"


# ---------------------------------------------------------
# LOAD MODEL + LABELS
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    # Load model pipeline
    pipe = joblib.load(MODEL_PATH)

    # Load labels.json
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # labels.json might be a dict OR list
    if isinstance(raw, dict):
        inv = {int(k): v for k, v in raw.items()}
    elif isinstance(raw, list):
        inv = {i: v for i, v in enumerate(raw)}
    else:
        raise ValueError("labels.json must be dict or list.")

    return pipe, inv


pipe, INV_LABELS = load_pipeline()


# ---------------------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------------------
def predict_top(pipe, inv_labels, ingredients, k=3):
    text = " ".join([i.strip().lower() for i in ingredients])
    proba = pipe.predict_proba([text])[0]
    idx = np.argsort(proba)[::-1][:k]
    return [(inv_labels[int(i)], float(proba[i])) for i in idx]


# ---------------------------------------------------------
# RECIPE GENERATOR (simple deterministic instructions)
# ---------------------------------------------------------
def generate_recipe(ingredients, cuisine):
    clean = [i.strip() for i in ingredients if i.strip()]
    title = f"{cuisine.title()} Style Dish with {clean[0]}" if clean else f"{cuisine.title()} Style Dish"

    steps = [
        "Chop aromatics such as garlic, onion, or ginger.",
        "Season main ingredients with salt, pepper, and spices.",
        "Heat oil in a pan and sauté aromatics lightly.",
        f"Add the main ingredients ({', '.join(clean)}) and stir-fry until cooked.",
        "Add herbs or sauce and simmer for 3–5 minutes.",
        "Taste, adjust seasoning, and serve warm."
    ]
    return title, steps


# ---------------------------------------------------------
# UNSPLASH IMAGE FETCHER (no API key needed)
# ---------------------------------------------------------
def get_cuisine_image(cuisine):
    try:
        q = requests.utils.quote(f"{cuisine} food dish")
        return f"https://source.unsplash.com/800x600/?{q}"
    except:
        return None


# ---------------------------------------------------------
# TTS SPEAKER
# ---------------------------------------------------------
def speak(text, lang="en"):
    buf = io.BytesIO()
    gTTS(text, lang=lang).write_to_fp(buf)
    buf.seek(0)
    return buf


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Smart Recipe Finder", page_icon="🍽️")
st.title("🍽️ Smart Recipe Finder")
st.caption("Predict cuisine • Generate simple recipe • Text-to-speech")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Settings")
    do_tts = st.checkbox("Enable Voice (English TTS)", value=False)
    st.markdown("---")
    st.write("**Model:** Logistic Regression + TF-IDF")
    st.write("**Classes:** 20")


# ---------------- INPUT ----------------
with st.expander("Enter Ingredients", expanded=True):
    raw_text = st.text_area(
        "Comma-separated ingredients:",
        "chicken, soy sauce, ginger, garlic, sesame oil",
        height=120,
    )
    ingredients = [x.strip() for x in raw_text.split(",") if x.strip()]


# ---------------- PREDICT BUTTON ----------------
if st.button("Predict Cuisine", type="primary"):
    if not ingredients:
        st.error("Please enter at least one ingredient.")
        st.stop()

    # ---- Predict ----
    top3 = predict_top(pipe, INV_LABELS, ingredients, k=3)
    top1, top1_prob = top3[0]

    # ---- Chart ----
    st.subheader("Predicted Cuisine (Top 3)")
    st.bar_chart(
        {"probability": [p for _, p in top3]},
        x=None,
        height=300
    )

    # ---- Image ----
    st.subheader(f"Image • {top1.title()}")
    img = get_cuisine_image(top1)
    if img:
        st.image(img, use_column_width=True)
    else:
        st.info("No image available.")

    # ---- Recipe ----
    title, steps = generate_recipe(ingredients, top1)
    st.markdown(f"### 🍽️ {title}")

    recipe_text = title + "\n\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
    st.text_area("Generated Recipe", recipe_text, height=220)

    # ---- TTS ----
    if do_tts:
        audio = speak(recipe_text, "en")
        st.audio(audio, format="audio/mp3")
